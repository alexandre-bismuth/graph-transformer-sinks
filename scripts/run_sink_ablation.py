"""Inference-time sink ablation with matched controls.

Per test graph: identify the top sink node (head-averaged received attention, mean over
layers), then re-run inference with that node's key column zeroed + rows renormalised in
every layer, and compare against controls: matched random node, highest-degree node, and
second-strongest sink. Reports task metric and final-layer Dirichlet (per-edge) per
condition, paired per graph.

GPS: nn.MultiheadAttention is replaced by an explicit manual implementation using the
module's own parameters (equivalence-checked against the stock forward on the first
graph). GRIT: uses the built-in mha.ablate_node hook.

Usage: python scripts/run_sink_ablation.py --run-dir outputs/zinc-graphgps [--n-graphs 500]
       python scripts/run_sink_ablation.py --run-dir outputs/zinc-graphgps --no-renorm
--no-renorm: zero the column but do not renormalise the rows, i.e. delete
the node's value contribution while leaving every other weight unchanged. Renormalised
ablation asks "does the model need somewhere to put this attention?"; the un-renormalised
one asks "does this node's value carry content?". Output: sink_ablation_norenorm_s<seed>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.seed import set_seed
from src.metrics_v2 import compute_dirichlet_v2
from scripts.run_eval_suite import load_run_config, detect_arch, build_everything

_ABLATE_COL = {'col': None, 'renorm': True}


def _patch_gps_mha(model):
    """Replace each GPS layer's MultiheadAttention forward with a manual, ablatable one."""
    originals = []
    for gps_layer in model.layers:
        for module in gps_layer.modules():
            if isinstance(module, nn.MultiheadAttention):
                originals.append((module, module.forward))

                def make_fwd(mha):
                    def fwd(query, key, value, key_padding_mask=None, need_weights=False,
                            attn_mask=None, average_attn_weights=True, is_causal=False):
                        x = query                       # batch_first=True in GPSConv
                        B, N, D = x.shape
                        Hh = mha.num_heads
                        hd = D // Hh
                        qkv = x @ mha.in_proj_weight.t()
                        if mha.in_proj_bias is not None:
                            qkv = qkv + mha.in_proj_bias
                        q, k, v = qkv.split(D, dim=-1)
                        q = q.view(B, N, Hh, hd).transpose(1, 2)
                        k = k.view(B, N, Hh, hd).transpose(1, 2)
                        v = v.view(B, N, Hh, hd).transpose(1, 2)
                        logits = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
                        if key_padding_mask is not None:
                            logits = logits.masked_fill(
                                key_padding_mask[:, None, None, :], float('-inf'))
                        w = F.softmax(logits, dim=-1)
                        col = _ABLATE_COL['col']
                        if col is not None:
                            w = w.clone()
                            w[..., col] = 0.0
                            if _ABLATE_COL['renorm']:
                                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        out = torch.matmul(w, v)
                        out = out.transpose(1, 2).reshape(B, N, D)
                        out = mha.out_proj(out)
                        # gps_conv calls with need_weights=False, but the instrumented
                        # model's hooks (used for sink identification) need the per-head
                        # weights every forward — mirror the stock wrapper, which forces
                        # need_weights=True / average_attn_weights=False.
                        return (out, w)
                    return fwd

                module.forward = make_fwd(module)
                break
    return originals


def _unpatch(originals):
    for module, fwd in originals:
        module.forward = fwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--n-graphs', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--no-renorm', action='store_true')
    args = p.parse_args()
    _ABLATE_COL['renorm'] = not args.no_renorm

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_run_config(args.run_dir)
    arch = detect_arch(config)
    task = config['architecture']['task']
    set_seed(args.seed)

    model, test_loader, _, _ = build_everything(config, arch, device)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model._register_attn_hooks()

    dataset = test_loader.dataset
    rng = np.random.RandomState(args.seed)
    n_graphs = min(args.n_graphs, len(dataset))

    from torch_geometric.data import Batch

    def set_ablation(col):
        if arch == 'gps':
            _ABLATE_COL['col'] = col
        else:
            for layer in model.layers:
                layer.mha.ablate_node = col
                layer.mha.ablate_renorm = not args.no_renorm

    originals = _patch_gps_mha(model) if arch == 'gps' else []

    # equivalence check (GPS): manual forward must reproduce stock predictions
    if arch == 'gps':
        data0 = dataset[0]
        b0 = Batch.from_data_list([data0]).to(device)
        set_ablation(None)
        with torch.no_grad():
            pred_manual = model(b0, collect_diagnostics=False)
        _unpatch(originals)
        with torch.no_grad():
            pred_stock = model(b0, collect_diagnostics=False)
        diff = (pred_manual - pred_stock).abs().max().item()
        print(f"manual-vs-stock max|diff| = {diff:.2e}")
        assert diff < 1e-3, "manual attention does not match stock forward"
        originals = _patch_gps_mha(model)

    conds = ['none', 'sink', 'random', 'topdeg', 'sink2']
    preds = {c: [] for c in conds}
    dirich = {c: [] for c in conds}
    targets = []

    with torch.no_grad():
        for gi in range(n_graphs):
            data = dataset[gi]
            n = data.num_nodes
            if n < 4:
                continue
            batch = Batch.from_data_list([data]).to(device)

            # identify sinks from a clean diagnostics pass
            set_ablation(None)
            model(batch, collect_diagnostics=True)
            scores = torch.zeros(n)
            for aw in model.attn_weights:
                if aw is not None:
                    scores += aw[0].mean(dim=1).mean(dim=0)[:n]
            order = torch.argsort(scores, descending=True)
            sink, sink2 = int(order[0]), int(order[1])
            deg = torch.bincount(data.edge_index[0], minlength=n)
            topdeg = int(torch.argmax(deg))
            cand = [i for i in range(n) if i != sink]
            rand = int(rng.choice(cand))
            node_for = {'none': None, 'sink': sink, 'random': rand,
                        'topdeg': topdeg, 'sink2': sink2}

            local_ei = data.edge_index

            for c in conds:
                set_ablation(node_for[c])
                out = model(batch, collect_diagnostics=True)
                preds[c].append(out.detach().cpu())
                Hf = model.layer_data[-1]['h'][:n].float()
                dirich[c].append(compute_dirichlet_v2(Hf, local_ei)['dirichlet_per_edge'])
            targets.append(data.y.view(-1).float())
            set_ablation(None)

    if arch == 'gps':
        _unpatch(originals)
    else:
        set_ablation(None)
    model._remove_attn_hooks()

    y = torch.cat(targets)
    result = {'n_graphs': len(targets), 'renorm': not args.no_renorm, 'conditions': {}}
    for c in conds:
        ph = torch.cat([p_.view(-1) for p_ in preds[c]])
        if task == 'regression':
            metric = (ph - y).abs().mean().item()
            mname = 'mae'
        else:
            metric = float('nan')
            mname = 'na'
        result['conditions'][c] = {
            mname: metric,
            'dirichlet_per_edge_final': float(np.mean(dirich[c])),
        }
    tag = 'sink_ablation_norenorm' if args.no_renorm else 'sink_ablation'
    path = os.path.join(args.run_dir, f'{tag}_s{args.seed}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result['conditions'], indent=1))
    print(f"saved {path}")


if __name__ == '__main__':
    main()
