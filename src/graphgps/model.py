import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, ResGatedGraphConv, global_add_pool, MessagePassing


class IdentityConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr=None)
        self.channels = channels

    def forward(self, x, edge_index, **kwargs):
        return x

class InstrumentedGPS(nn.Module):
    """GraphGPS with hooks to collect per-layer attention weights and hidden states."""

    def __init__(self, config, dataset_info):
        super().__init__()

        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        attn_dropout = config['model']['attn_dropout']
        dropout = config['model']['dropout']
        num_layers = config['architecture']['num_layers']
        mpnn_type = config['architecture']['mpnn']
        task = config['architecture']['task']
        pe_type = config['pe']['type']
        pe_dim = config['pe']['dim'] if pe_type != 'none' else 0

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        self.pe_type = pe_type
        self.task = task
        self.input_type = dataset_info['input_type']

        feat_dim = hidden_dim - pe_dim

        if self.input_type == 'categorical':
            num_node_types = dataset_info['num_node_types']
            num_edge_types = dataset_info['num_edge_types']
            if config['vnode']['enabled']:
                num_node_types = config['vnode']['num_node_types']
                num_edge_types = config['vnode']['num_edge_types']
            self.node_emb = nn.Embedding(num_node_types, feat_dim)
            self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        else:
            node_feat_dim = dataset_info['node_feat_dim']
            edge_feat_dim = dataset_info['edge_feat_dim']
            self.node_emb = nn.Linear(node_feat_dim, feat_dim)
            self.edge_emb = nn.Linear(edge_feat_dim, hidden_dim)

        if pe_dim > 0:
            self.pe_encoder = nn.Sequential(
                nn.BatchNorm1d(pe_dim),
                nn.Linear(pe_dim, pe_dim),
                nn.ReLU(),
                nn.Linear(pe_dim, pe_dim),
            )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if mpnn_type == 'gine':
                gine_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                local_model = GINEConv(gine_nn, edge_dim=hidden_dim)
            elif mpnn_type == 'gatedgcn':
                local_model = ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            else:
                local_model = IdentityConv(hidden_dim)

            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_model,
                heads=num_heads,
                dropout=dropout,
                norm='batch_norm',
                attn_type='multihead',
                attn_kwargs={'dropout': attn_dropout},
            )
            self.layers.append(gps_layer)

        num_classes = dataset_info['num_classes']
        out_dim = 1 if task == 'regression' else num_classes
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self.layer_data = []
        self.attn_weights = []
        self._attn_hooks = []

    def _register_attn_hooks(self):
        self._remove_attn_hooks()
        self.attn_weights = [None] * self.num_layers

        for layer_idx, gps_layer in enumerate(self.layers):
            for name, module in gps_layer.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    original_forward = module.forward
                    def patched_forward(orig_fwd=original_forward):
                        def wrapper(*args, **kwargs):
                            kwargs['need_weights'] = True
                            kwargs['average_attn_weights'] = False
                            return orig_fwd(*args, **kwargs)
                        return wrapper
                    module.forward = patched_forward()

                    def make_hook(idx):
                        def hook_fn(module, args, output):
                            if isinstance(output, tuple) and len(output) == 2 \
                                    and output[1] is not None:
                                self.attn_weights[idx] = output[1].detach().cpu()
                        return hook_fn

                    handle = module.register_forward_hook(make_hook(layer_idx))
                    self._attn_hooks.append(handle)
                    break

    def _remove_attn_hooks(self):
        for h in self._attn_hooks:
            h.remove()
        self._attn_hooks = []
        self.attn_weights = []

    @torch.no_grad()
    def compute_logits_values(self, layer_idx, x_graph):
        """Offline pre-softmax logits and per-node value norms for one graph.

        x_graph: (n, D) input to layer `layer_idx` (= layer_data[layer_idx]['h'] slice).
        Returns logits (H, n, n) and value norms (H, n), both on CPU. Uses the layer's own
        nn.MultiheadAttention parameters; GPSConv feeds the raw layer input to attention.
        """
        mha = None
        for module in self.layers[layer_idx].modules():
            if isinstance(module, nn.MultiheadAttention):
                mha = module
                break
        if mha is None:
            raise RuntimeError('no MultiheadAttention in layer')
        x = x_graph.float().to(mha.in_proj_weight.device)
        n, D = x.shape
        H = mha.num_heads
        hd = D // H
        qkv = x @ mha.in_proj_weight.t()
        if mha.in_proj_bias is not None:
            qkv = qkv + mha.in_proj_bias
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(n, H, hd).permute(1, 0, 2)
        k = k.view(n, H, hd).permute(1, 0, 2)
        v = v.view(n, H, hd).permute(1, 0, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        return logits.cpu(), v.norm(dim=-1).cpu()

    def forward(self, batch_data, collect_diagnostics=False):
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        if self.input_type == 'categorical':
            x = self.node_emb(x.squeeze(-1))
        else:
            if x.dtype == torch.long:
                x = x.float()
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            x = self.node_emb(x)

        if self.pe_dim > 0:
            pe_attr = None
            if self.pe_type == 'rwse' and hasattr(batch_data, 'random_walk_pe'):
                pe_attr = batch_data.random_walk_pe
            elif self.pe_type == 'lappe' and hasattr(batch_data, 'laplacian_pe'):
                pe_attr = batch_data.laplacian_pe

            if pe_attr is not None:
                pe = self.pe_encoder(pe_attr)
                x = torch.cat([x, pe], dim=-1)
            else:
                x = F.pad(x, (0, self.pe_dim))

        if self.input_type == 'categorical':
            edge_attr = self.edge_emb(edge_attr.squeeze(-1))
        else:
            if edge_attr.dtype == torch.long:
                edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.edge_emb(edge_attr)

        if collect_diagnostics:
            self.layer_data = [{'h': x.detach().cpu(), 'batch': batch.detach().cpu()}]

        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr)
            if collect_diagnostics:
                self.layer_data.append({'h': x.detach().cpu(), 'batch': batch.detach().cpu()})

        if self.task == 'node_classification':
            return self.output_head(x)
        else:
            graph_emb = global_add_pool(x, batch)
            out = self.output_head(graph_emb)
            if self.task == 'regression':
                return out.squeeze(-1)
            return out
