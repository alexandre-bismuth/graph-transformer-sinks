import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree, to_dense_batch

def signed_sqrt(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))


class GritMultiHeadAttention(nn.Module):
    """GRIT attention with a switchable scoring path.

    attn_mode:
      'grit'     — signed-sqrt edge-conditioned scoring (default). Clamps each
                   coordinate of the score to [-5, 5] before the ReLU, so e_hat in [0, 5]^h.
      'dot_bias' — scaled dot-product logits + additive edge/RRWP bias (Graphormer-style);
                   plain values (no edge-value channel)
      'dot'      — scaled dot-product only; no edge conditioning anywhere

    Scoring dial (all share the 'grit' parameter set, so a 'grit'
    checkpoint loads into any of them):
      'grit_dotlogit' — GRIT with only the logit replaced: logit = q.k/sqrt(h). e_hat is
                        still computed as in GRIT and still feeds the values (VeRow) and
                        the next layer's edge stream. W_A exists but is unused.
      'grit_plusdot'  — logit = W_A(e_hat) + q.k/sqrt(h): GRIT's score plus an unbounded
                        dot-product term.
      'grit_noclamp'  — GRIT without any clamp: e_hat = ReLU(ssq(.) + b), unbounded.
      'grit_official' — clamp placement of the official GRIT code (LiamMa/GRIT): e_hat
                        unclipped, scalar logit clamped to [-5, 5] after the readout.
    """

    GRIT_FAMILY = ('grit', 'grit_dotlogit', 'grit_plusdot', 'grit_noclamp', 'grit_official')

    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0, attn_mode='grit'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_mode = attn_mode
        assert attn_mode in self.GRIT_FAMILY + ('dot_bias', 'dot'), \
            f'unknown attn_mode {attn_mode!r}'
        h = self.head_dim

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        if attn_mode in self.GRIT_FAMILY:
            self.W_E = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
            self.W_VeRow = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.W_A = nn.Linear(h, 1, bias=False)
            self.W_Eo = nn.Linear(hidden_dim, hidden_dim, bias=True)
        elif attn_mode == 'dot_bias':
            self.W_Eb = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.W_A = nn.Linear(h, 1, bias=False)
            self.W_Eo = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # diagnostics / interventions (all off by default; no state)
        self.collect_extra = False       # stash logits / value norms / clamp frac
        self.entropy_track = False       # accumulate row-entropy (entropy forcing)
        self.ablate_node = None          # sink ablation: zero this key column + renormalise
        self.ablate_renorm = True        # False: zero the column without renormalising rows
        self._attn_logits = None
        self._value_norms = None
        self._clamp_frac = None          # scalar, whole batch
        self._sat_stats = None           # dict name -> (B,) per-graph saturation fractions
        self._row_entropy = None

    def forward(self, x_dense, e, node_mask, vnode_cols=None, sink_bias=None):
        B, N, D = x_dense.shape
        H = self.num_heads
        h = self.head_dim

        q = self.W_Q(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)
        k = self.W_K(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)
        v = self.W_V(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)

        e_hat = None
        if self.attn_mode in self.GRIT_FAMILY:
            pair_valid = None
            if self.collect_extra:
                pair_valid = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))   # (B, N, N)
            qk = q.unsqueeze(3) + k.unsqueeze(2)
            e_proj = self.W_E(e).reshape(B, N, N, H, h * 2).permute(0, 3, 1, 2, 4)
            e_w = e_proj[..., :h]
            e_b = e_proj[..., h:]
            score = signed_sqrt(qk * e_w) + e_b
            if self.collect_extra:
                # pre-clamp score coordinates: >= 5 hits the ceiling (e_hat saturates at 5);
                # <= 0 is zeroed by the ReLU whatever the clamp does.
                f = lambda t: self._masked_graph_mean(t.float().mean(dim=-1), pair_valid)
                self._sat_stats = {'clamp_upper_frac': f(score >= 5.0),
                                   'clamp_lower_frac': f(score <= -5.0),
                                   'relu_zero_frac': f(score <= 0.0)}
            if self.attn_mode in ('grit', 'grit_dotlogit', 'grit_plusdot'):
                if self.collect_extra:
                    self._clamp_frac = (score.abs() >= 5.0).float().mean().item()
                    self._sat_stats['clamp_frac'] = self._masked_graph_mean(
                        (score.abs() >= 5.0).float().mean(dim=-1), pair_valid)
                score = torch.clamp(score, -5.0, 5.0)
            e_hat = F.relu(score)
            if self.attn_mode == 'grit_dotlogit':
                attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (h ** 0.5)
            else:
                attn_logits = self.W_A(e_hat).squeeze(-1)
                if self.attn_mode == 'grit_plusdot':
                    attn_logits = attn_logits + torch.matmul(q, k.transpose(-2, -1)) / (h ** 0.5)
                elif self.attn_mode == 'grit_official':
                    if self.collect_extra:
                        hit = (attn_logits.abs() >= 5.0).float()                 # (B, H, N, N)
                        self._clamp_frac = hit.mean().item()
                        self._sat_stats['logit_clamp_frac'] = self._masked_graph_mean(hit, pair_valid)
                    attn_logits = torch.clamp(attn_logits, -5.0, 5.0)
        elif self.attn_mode == 'dot_bias':
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (h ** 0.5)
            e_b = self.W_Eb(e).reshape(B, N, N, H, h).permute(0, 3, 1, 2, 4)
            e_hat = F.relu(torch.clamp(e_b, -5.0, 5.0))
            attn_logits = attn_logits + self.W_A(e_hat).squeeze(-1)
        else:  # 'dot'
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (h ** 0.5)

        if sink_bias is not None and vnode_cols is not None:
            # bias forcing: add a learned scalar to every logit pointing at the vnode
            col_onehot = F.one_hot(vnode_cols, num_classes=N).to(attn_logits.dtype)  # (B, N)
            attn_logits = attn_logits + sink_bias * col_onehot.unsqueeze(1).unsqueeze(1)

        attn_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        attn_logits = attn_logits.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights.masked_fill(~attn_mask.unsqueeze(1), 0.0)

        if self.ablate_node is not None:
            w = attn_weights.clone()
            w[:, :, :, self.ablate_node] = 0.0
            if self.ablate_renorm:
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_weights = w.masked_fill(~attn_mask.unsqueeze(1), 0.0)

        if self.entropy_track:
            p = attn_weights.clamp_min(1e-10)
            row_ent = -(attn_weights * p.log()).sum(dim=-1)          # (B, H, N)
            valid = node_mask.unsqueeze(1).expand(-1, H, -1)
            self._row_entropy = (row_ent * valid).sum() / valid.sum().clamp_min(1)

        if self.collect_extra:
            self._attn_logits = attn_logits.detach().cpu()
            self._value_norms = v.norm(dim=-1).detach().cpu()        # (B, H, N)

        attn_weights = self.attn_dropout(attn_weights)

        x_V = torch.matmul(attn_weights.unsqueeze(-2), v.unsqueeze(2)).squeeze(-2)
        if self.attn_mode in self.GRIT_FAMILY:
            e_weighted = attn_weights.unsqueeze(-1) * e_hat
            e_sum = e_weighted.sum(dim=3)
            x_attn = x_V + self.W_VeRow(e_sum.permute(0, 2, 1, 3).reshape(B, N, D)).reshape(B, N, H, h).permute(0, 2, 1, 3)
        else:
            x_attn = x_V

        x_out = self.W_O(x_attn.permute(0, 2, 1, 3).reshape(B, N, D))
        e_out = None
        if self.attn_mode in self.GRIT_FAMILY + ('dot_bias',):
            e_out = self.W_Eo(e_hat.permute(0, 2, 3, 1, 4).reshape(B, N, N, D))

        return x_out, e_out, attn_weights

    @staticmethod
    def _masked_graph_mean(hit, pair_valid):
        """hit: (B, H, N, N) in [0, 1]; pair_valid: (B, N, N) bool -> (B,) cpu tensor."""
        if pair_valid is None:
            return None
        m = pair_valid.unsqueeze(1).to(hit.dtype)
        return ((hit * m).sum(dim=(1, 2, 3)) /
                (m.sum(dim=(1, 2, 3)) * hit.shape[1]).clamp_min(1)).detach().cpu()


class GritTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0, dropout=0.0, attn_mode='grit'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.mha = GritMultiHeadAttention(hidden_dim, num_heads, attn_dropout, attn_mode=attn_mode)

        self.deg_scaler_1 = nn.Parameter(torch.ones(hidden_dim))
        self.deg_scaler_2 = nn.Parameter(torch.zeros(hidden_dim))

        self.bn_node_attn = nn.BatchNorm1d(hidden_dim)
        if attn_mode in GritMultiHeadAttention.GRIT_FAMILY + ('dot_bias',):
            self.bn_edge = nn.BatchNorm1d(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.bn_node_ffn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x_dense, e, node_mask, log_deg_dense, vnode_cols=None, sink_bias=None):
        B, N, D = x_dense.shape

        x_attn, e_attn, attn_weights = self.mha(x_dense, e, node_mask,
                                                vnode_cols=vnode_cols, sink_bias=sink_bias)

        x_attn = x_attn * self.deg_scaler_1 + log_deg_dense * x_attn * self.deg_scaler_2
        x_dense = x_dense + self.bn_node_attn(x_attn.reshape(-1, D)).reshape(B, N, D)

        if e_attn is not None:
            e = e + self.bn_edge(e_attn.reshape(-1, D)).reshape(B, N, N, D)

        x_ffn = self.ffn(x_dense.reshape(-1, D))
        x_dense = x_dense + self.bn_node_ffn(x_ffn).reshape(B, N, D)

        self._attn_weights = attn_weights

        return x_dense, e


class InstrumentedGRIT(nn.Module):
    """GRIT with hooks to collect per-layer attention weights and hidden states."""

    def __init__(self, config, dataset_info):
        super().__init__()

        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        attn_dropout = config['model']['attn_dropout']
        dropout = config['model']['dropout']
        num_layers = config['architecture']['num_layers']
        task = config['architecture']['task']
        pe_dim = config['pe']['dim']

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task = task
        self.pe_dim = pe_dim
        self.input_type = dataset_info['input_type']
        self.pool = config.get('model', {}).get('pool', 'mean')
        self.attn_mode = config.get('model', {}).get('attn_mode', 'grit')
        self.use_node_pe = config.get('model', {}).get('use_node_pe', True)
        # 'full' (default): bond-type embeddings + RRWP pair encodings initialise the pair
        # stream. 'none': pair stream starts at zero (GRIT-blind control, information-matched
        # to the 'dot' arm, which sees neither bonds nor pair encodings).
        self.edge_input = config.get('model', {}).get('edge_input', 'full')
        assert self.edge_input in ('full', 'none'), self.edge_input
        self._has_pair_stream = self.attn_mode in GritMultiHeadAttention.GRIT_FAMILY + ('dot_bias',)
        self.has_vnode = config.get('vnode', {}).get('enabled', False)
        use_sink_bias = config.get('model', {}).get('sink_bias', False)
        self.sink_bias_param = nn.Parameter(torch.zeros(())) if use_sink_bias else None

        if self.input_type == 'categorical':
            num_node_types = dataset_info['num_node_types']
            num_edge_types = dataset_info['num_edge_types']
            self.node_emb = nn.Embedding(num_node_types, hidden_dim)
            self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        else:
            node_feat_dim = dataset_info['node_feat_dim']
            edge_feat_dim = dataset_info['edge_feat_dim']
            self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
            self.edge_emb = nn.Linear(edge_feat_dim, hidden_dim)

        if self.use_node_pe:
            self.pe_node_enc = nn.Linear(pe_dim, hidden_dim)
        if self._has_pair_stream and self.edge_input == 'full':
            self.pe_edge_enc = nn.Linear(pe_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GritTransformerLayer(hidden_dim, num_heads, attn_dropout, dropout,
                                 attn_mode=self.attn_mode)
            for _ in range(num_layers)
        ])

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
        self.attn_logits = []
        self.value_norms = []
        self.clamp_fracs = []          # per layer: dict name -> (B,) saturation fractions, or None

    def pop_attn_entropy(self):
        """Mean attention row-entropy across layers from the last forward (with grad)."""
        ents = [l.mha._row_entropy for l in self.layers if l.mha._row_entropy is not None]
        for l in self.layers:
            l.mha._row_entropy = None
        if not ents:
            return None
        return torch.stack(ents).mean()

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

        if self.use_node_pe and hasattr(batch_data, 'rrwp_node') and batch_data.rrwp_node is not None:
            x = x + self.pe_node_enc(batch_data.rrwp_node)

        x_dense, node_mask = to_dense_batch(x, batch)
        B, N_max, D = x_dense.shape

        e_dense = None
        if self._has_pair_stream:
            e_dense = torch.zeros(B, N_max, N_max, self.hidden_dim, device=x.device)

            if edge_attr is not None and self.edge_input == 'full':
                if self.input_type == 'categorical':
                    ea = self.edge_emb(edge_attr.squeeze(-1))
                else:
                    if edge_attr.dtype == torch.long:
                        edge_attr = edge_attr.float()
                    if edge_attr.dim() == 1:
                        edge_attr = edge_attr.unsqueeze(-1)
                    ea = self.edge_emb(edge_attr)

                src_global, dst_global = edge_index[0], edge_index[1]
                edge_batch = batch[src_global]
                _, counts = torch.unique_consecutive(batch, return_counts=True)
                offsets = torch.zeros(B, device=batch.device, dtype=torch.long)
                offsets[1:] = counts.cumsum(0)[:-1]
                src_local = src_global - offsets[edge_batch]
                dst_local = dst_global - offsets[edge_batch]
                e_dense[edge_batch, src_local, dst_local] = ea

            if self.edge_input == 'full' and hasattr(batch_data, 'rrwp_edge') \
                    and batch_data.rrwp_edge is not None:
                rrwp_pe = self.pe_edge_enc(batch_data.rrwp_edge)
                _, counts = torch.unique_consecutive(batch, return_counts=True)
                offset = 0
                for b_idx in range(B):
                    n = counts[b_idx].item()
                    e_dense[b_idx, :n, :n] = e_dense[b_idx, :n, :n] + rrwp_pe[offset:offset + n * n].reshape(n, n, self.hidden_dim)
                    offset += n * n

        deg = degree(edge_index[0], num_nodes=x.size(0)).float()
        log_deg = torch.log(1.0 + deg)
        log_deg_dense, _ = to_dense_batch(log_deg.unsqueeze(-1), batch)

        vnode_cols = None
        if self.sink_bias_param is not None and self.has_vnode:
            vnode_cols = node_mask.sum(dim=1) - 1     # vnode is the last real node per graph

        if collect_diagnostics:
            self.layer_data = [{'h': x.detach().cpu(), 'batch': batch.detach().cpu()}]
            self.attn_weights = []
            self.attn_logits = []
            self.value_norms = []
            self.clamp_fracs = []
            for layer in self.layers:
                layer.mha.collect_extra = True
                layer.mha._sat_stats = None

        for layer_idx, layer in enumerate(self.layers):
            x_dense, e_dense = layer(x_dense, e_dense, node_mask, log_deg_dense,
                                     vnode_cols=vnode_cols, sink_bias=self.sink_bias_param)
            if collect_diagnostics:
                x_flat = x_dense[node_mask]
                self.layer_data.append({'h': x_flat.detach().cpu(), 'batch': batch.detach().cpu()})
                self.attn_weights.append(layer._attn_weights.detach().cpu())
                self.attn_logits.append(layer.mha._attn_logits)
                self.value_norms.append(layer.mha._value_norms)
                self.clamp_fracs.append(layer.mha._sat_stats)

        if collect_diagnostics:
            for layer in self.layers:
                layer.mha.collect_extra = False

        x_flat = x_dense[node_mask]

        if self.task == 'node_classification':
            return self.output_head(x_flat)
        else:
            if self.pool == 'sum':
                graph_emb = global_add_pool(x_flat, batch)
            else:
                graph_emb = global_mean_pool(x_flat, batch)
            out = self.output_head(graph_emb)
            if self.task == 'regression':
                return out.squeeze(-1)
            return out

    def _register_attn_hooks(self):
        pass

    def _remove_attn_hooks(self):
        pass
