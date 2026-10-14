"""Norm intervention: can a node win attention by growing its norm?

At layer l, the input hidden state of one uniformly chosen real node per graph is
multiplied by rho before the layer's attention is computed; everything else is left as
the trained model produces it. We read that layer's attention only and record the
scaled node's received attention s_j = mean_u A_uj (head mean), n * s_j, the fraction of
heads with s_j > eps, and whether it becomes the argmax node. rho = 1 is the unmodified
model. The chosen node is fixed per graph (seeded), so every rho sees the same node.

Prediction from the scoring forms: a dot-product logit q_u . k_j grows linearly in rho,
so (when aligned) the node's share can approach 1; GRIT's saturating score keeps its
logit inside a fixed range, so its share is capped whatever rho is.

Works for GRIT-family runs (layer input x_dense, dense (B, N, D)) and GraphGPS runs
(layer input x, flat (num_nodes, D)).

Usage:
  python scripts/run_norm_intervention.py --run-dir outputs/zinc-grit-s0 [--layers 3 5 8]
Writes <run_dir>/norm_intervention.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_eval_suite import load_run_config, detect_arch, build_everything

EPS = 0.3
RHOS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default='best_model.pt')
    ap.add_argument('--layers', type=int, nargs='+', default=[3, 5, 8], help='1-based')
    ap.add_argument('--rhos', type=float, nargs='+', default=list(RHOS))
    ap.add_argument('--n-graphs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_run_config(args.run_dir)
    arch = detect_arch(config)
    has_vnode = config.get('vnode', {}).get('enabled', False)
    model, test_loader, _, _ = build_everything(config, arch, device)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    rng = np.random.default_rng(args.seed)

    # Fixed batches and fixed target node per graph (same for every rho and layer).
    batches, targets = [], []
    seen = 0
    for batch in test_loader:
        if seen >= args.n_graphs:
            break
        counts = torch.bincount(batch.batch).tolist()
        keep = min(len(counts), args.n_graphs - seen)
        if keep < len(counts):
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(batch.to_data_list()[:keep])
            counts = counts[:keep]
        tg = [int(rng.integers(0, c - 1 if has_vnode else c)) for c in counts]
        batches.append(batch)
        targets.append((counts, tg))
        seen += keep

    state_ = {'rho': 1.0, 'counts': None, 'tg': None}

    def hook_grit(module, args_, kwargs):
        x = args_[0].clone()
        for b, j in enumerate(state_['tg']):
            x[b, j] = x[b, j] * state_['rho']
        return (x,) + tuple(args_[1:]), kwargs

    def hook_gps(module, args_, kwargs):
        x = args_[0].clone()
        off = 0
        for c, j in zip(state_['counts'], state_['tg']):
            x[off + j] = x[off + j] * state_['rho']
            off += c
        return (x,) + tuple(args_[1:]), kwargs

    out = {'run': args.run_dir, 'arch': arch,
           'attn_mode': config.get('model', {}).get('attn_mode', 'gps' if arch == 'gps' else 'grit'),
           'eps': EPS, 'n_graphs': seen, 'seed': args.seed, 'layers': {}}
    for l in args.layers:
        assert 1 <= l <= model.num_layers
        layer = model.layers[l - 1]
        h = layer.register_forward_pre_hook(hook_grit if arch == 'grit' else hook_gps,
                                            with_kwargs=True)
        per_rho = {}
        for rho in args.rhos:
            state_['rho'] = rho
            s_all, ns_all, hf_all, top_all, norm_ratio = [], [], [], [], []
            with torch.no_grad():
                for batch, (counts, tg) in zip(batches, targets):
                    state_['counts'], state_['tg'] = counts, tg
                    batch = batch.to(device)
                    model._register_attn_hooks()
                    _ = model(batch, collect_diagnostics=True)
                    aw = model.attn_weights[l - 1]                # (B, H, N, N) dense, padded
                    hin = model.layer_data[l - 1]['h']            # input to layer l (pre-hook)
                    off = 0
                    for b, (c, j) in enumerate(zip(counts, tg)):
                        A = aw[b, :, :c, :c].float()
                        s = A.mean(dim=1)                         # (H, c)
                        sm = s.mean(dim=0)
                        s_all.append(float(sm[j]))
                        ns_all.append(float(c * sm[j]))
                        hf_all.append(float((s[:, j] > EPS).float().mean()))
                        top_all.append(float(int(sm.argmax()) == j))
                        hn = hin[off:off + c].norm(dim=-1)
                        norm_ratio.append(float(rho * hn[j] / hn.mean()))
                        off += c
                    model._remove_attn_hooks()
            q = lambda v, p: float(np.percentile(v, p))
            per_rho[str(rho)] = {
                's_mean': float(np.mean(s_all)), 's_median': q(s_all, 50), 's_p90': q(s_all, 90),
                'ns_median': q(ns_all, 50), 'ns_p90': q(ns_all, 90), 'ns_max': float(np.max(ns_all)),
                'frac_heads_above_eps_mean': float(np.mean(hf_all)),
                'frac_graphs_any_head_above_eps': float(np.mean([v > 0 for v in hf_all])),
                'frac_becomes_argmax': float(np.mean(top_all)),
                'norm_ratio_median': q(norm_ratio, 50),
            }
            r = per_rho[str(rho)]
            print(f"layer {l} rho {rho:>5}: n*s med {r['ns_median']:.2f} p90 {r['ns_p90']:.2f} "
                  f"max {r['ns_max']:.2f} | heads>eps {r['frac_heads_above_eps_mean']:.3f} "
                  f"| argmax {r['frac_becomes_argmax']:.3f} | ||h||/mean {r['norm_ratio_median']:.2f}")
        h.remove()
        out['layers'][str(l)] = per_rho
    path = args.out or os.path.join(args.run_dir, 'norm_intervention.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
