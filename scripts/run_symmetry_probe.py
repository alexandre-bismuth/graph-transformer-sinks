"""Vertex-transitive probe of the symmetry barrier.

Feeds uniform-labeled cycles C_n through trained checkpoints. On a vertex-transitive
labeled input the symmetry--locality theorem forces s_j = 1/n for every node in any
permutation-equivariant model (all-orbit size n). Models with sign-ambiguous LapPE are
not equivariant on the actual input and are free to concentrate.

Reports, per run and n: max_j s_j, max deviation |s_j - 1/n|, worst over layers/heads.
Saves <run>/symmetry_probe.json.

Virtual-node runs (config vnode.enabled): the cycle is augmented with the same
AddVirtualNode transform the model saw in training, in the training order
('recomputed' mode inserts the vnode before PE so it gets a real PE identity; default
'padded' mode appends it after with a zero PE row). On the augmented graph the vnode is
a singleton orbit — the barrier permits s_vnode up to 1 — while the n cycle nodes remain
one orbit, so the theorem still forces them to share a single value. Reported keys change
meaning accordingly: `max_abs_deviation` = worst within-orbit spread of the real nodes
(the forced-to-zero quantity), `s_vnode` = worst received attention on the vnode (the
sink-eligibility question), `max_s` = max of the two columns' worsts.
"""
import argparse
import json
import os
import sys

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_eval_suite import load_run_config, detect_arch, build_everything


def make_cycle(n):
    src = list(range(n)) + [(i + 1) % n for i in range(n)]
    dst = [(i + 1) % n for i in range(n)] + list(range(n))
    return Data(x=torch.zeros(n, 1, dtype=torch.long),
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=torch.ones(2 * n, dtype=torch.long),
                y=torch.zeros(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--sizes', default='8,12,16,20,24')
    p.add_argument('--device', default=None)
    p.add_argument('--fp64', action='store_true',
                   help='run in float64 (separates float32 rounding from equivariance violations); '
                        'writes symmetry_probe_fp64.json')
    args = p.parse_args()
    if args.fp64:
        torch.set_default_dtype(torch.float64)

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_run_config(args.run_dir)
    arch = detect_arch(config)
    model, _, _, _ = build_everything(config, arch, device)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    if args.fp64:
        model.double()
    model.eval()
    model._register_attn_hooks()

    if arch == 'gps':
        from src.graphgps.datasets import _build_pe_transform, _build_vnode_transform
    else:
        from src.datasets import _build_pe_transform, _build_vnode_transform
    pe_t = _build_pe_transform(config)
    vnode_cfg = config.get('vnode', {}) or {}
    vnode_t = _build_vnode_transform(config) if vnode_cfg.get('enabled', False) else None
    vnode_first = vnode_t is not None and vnode_cfg.get('pe_mode', 'padded') == 'recomputed'

    out = {'run_dir': args.run_dir, 'pe_type': config['pe']['type'],
           'vnode': vnode_t is not None, 'dtype': 'float64' if args.fp64 else 'float32',
           'vnode_pe_mode': (vnode_cfg.get('pe_mode', 'padded') if vnode_t is not None else None),
           'sizes': {}}
    with torch.no_grad():
        for n in [int(s) for s in args.sizes.split(',')]:
            g = make_cycle(n)
            if vnode_first:
                g = vnode_t(g)
            if pe_t is not None:
                g = pe_t(g)
            if vnode_t is not None and not vnode_first:
                g = vnode_t(g)
            if args.fp64:
                for key in g.keys():
                    if torch.is_tensor(g[key]) and g[key].is_floating_point():
                        g[key] = g[key].double()
            batch = Batch.from_data_list([g]).to(device)
            _ = model(batch, collect_diagnostics=True)
            if vnode_t is None:
                worst_dev, worst_smax = 0.0, 0.0
                for l in range(model.num_layers):
                    aw = model.attn_weights[l]
                    if aw is None:
                        continue
                    A = aw[0, :, :n, :n].float().cpu()   # (H, n, n)
                    s = A.mean(dim=1)                     # (H, n) column means
                    dev = (s - 1.0 / n).abs().max().item()
                    worst_dev = max(worst_dev, dev)
                    worst_smax = max(worst_smax, s.max().item())
                out['sizes'][n] = {'uniform_score': 1.0 / n,
                                   'max_s': worst_smax,
                                   'max_abs_deviation': worst_dev}
                print(f"n={n:3d} 1/n={1.0/n:.4f} max_s={worst_smax:.4f} max_dev={worst_dev:.2e}")
            else:
                N = n + 1                                 # AddVirtualNode appends at index n
                worst_dev, worst_real, worst_vn = 0.0, 0.0, 0.0
                for l in range(model.num_layers):
                    aw = model.attn_weights[l]
                    if aw is None:
                        continue
                    A = aw[0, :, :N, :N].float().cpu()   # (H, N, N)
                    s = A.mean(dim=1)                     # (H, N) received attention
                    real = s[:, :n]
                    # cycle nodes are still one orbit of the augmented graph: any spread
                    # around their per-head mean is an equivariance violation
                    dev = (real - real.mean(dim=1, keepdim=True)).abs().max().item()
                    worst_dev = max(worst_dev, dev)
                    worst_real = max(worst_real, real.max().item())
                    worst_vn = max(worst_vn, s[:, n].max().item())
                out['sizes'][n] = {'has_vnode': True,
                                   'uniform_score': 1.0 / N,
                                   'max_s': max(worst_real, worst_vn),
                                   'max_abs_deviation': worst_dev,
                                   's_vnode': worst_vn,
                                   'real_max_s': worst_real}
                print(f"n={n:3d} 1/N={1.0/N:.4f} s_vnode={worst_vn:.4f} "
                      f"real_max_s={worst_real:.4f} orbit_dev={worst_dev:.2e}")

    fn = os.path.join(args.run_dir, 'symmetry_probe_fp64.json' if args.fp64 else 'symmetry_probe.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', fn)


if __name__ == '__main__':
    main()
