"""Figure 1 data: measured per-atom received attention on one ZINC test molecule under
several trained models.

For every run and every layer l it records the per-head sink scores
s_j^(l,h) = mean_u A_uj^(l,h) (column mean of the row-stochastic attention), so
sum_j s_j^(l,h) = 1 exactly; the plot uses the head average at the run's peak-sinking
layer (argmax over layers of the run-level sink rate in eval_summary.json — the same
layer the paper's sink rate is read at).

Molecule selection is fixed and model-independent: the first test graph (dataset order,
shuffle=False) with exactly --n-atoms atoms (default 23 = median ZINC test size).

Usage:
  python scripts/make_fig1_data.py --runs outputs/zinc-grit-s0 outputs/zinc-grit-dot-s0 \
      outputs/zinc-graphgps-s0 --labels GRIT GRIT-dot GraphGPS -o outputs/figures/fig1_data.json
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


def peak_layer(run_dir):
    with open(os.path.join(run_dir, 'eval_summary.json')) as f:
        agg = json.load(f)['agg']
    m = agg['overall_sink_rate']['mean']
    vals = [(v, i) for i, v in enumerate(m) if v == v]
    return max(vals)[1]          # agg index == layer index (index 0 = input, NaN)


def find_graph(dataset, n_atoms, select):
    if select.startswith('index:'):
        return int(select.split(':')[1])
    for i in range(len(dataset)):
        if int(dataset[i].num_nodes) == n_atoms:
            return i
    raise ValueError(f'no test graph with {n_atoms} atoms')


@torch.no_grad()
def per_layer_scores(model, data, device):
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data]).to(device)
    model.eval()
    model._register_attn_hooks()
    _ = model(batch, collect_diagnostics=True)
    n = int(data.num_nodes)
    out = []
    for l in range(model.num_layers):
        aw = model.attn_weights[l]
        A = aw[0, :, :n, :n].float()                      # (H, n, n), rows sum to 1
        out.append(A.mean(dim=1).cpu().numpy())           # (H, n): s_j per head
    model._remove_attn_hooks()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--n-atoms', type=int, default=23)
    ap.add_argument('--select', default='first', help="'first' or 'index:K' (test-set index)")
    ap.add_argument('--ckpt', default='best_model.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('-o', '--out', default='outputs/figures/fig1_data.json')
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r) for r in args.runs]
    assert len(labels) == len(args.runs)
    device = torch.device(args.device)

    ref = None
    panels = []
    for run, lab in zip(args.runs, labels):
        config = load_run_config(run)
        if config.get('vnode', {}).get('enabled', False):
            raise ValueError(f'{run}: virtual-node arms are not supported in Figure 1')
        arch = detect_arch(config)
        model, test_loader, _, _ = build_everything(config, arch, device)
        state = torch.load(os.path.join(run, args.ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
        ds = test_loader.dataset
        gi = find_graph(ds, args.n_atoms, args.select)
        data = ds[gi]
        ident = {'test_index': gi, 'num_nodes': int(data.num_nodes),
                 'edge_index': data.edge_index.tolist(),
                 'atom_type': data.x.view(-1).tolist() if data.x.dim() == 1 or data.x.size(-1) == 1
                 else data.x[:, 0].tolist(),
                 'bond_type': (data.edge_attr.view(-1).tolist() if data.edge_attr is not None
                               and (data.edge_attr.dim() == 1 or data.edge_attr.size(-1) == 1)
                               else None),
                 'target': float(data.y.view(-1)[0])}
        if ref is None:
            ref = ident
        else:   # the same molecule in every panel (GRIT and GPS loaders are built separately)
            for k in ('test_index', 'num_nodes', 'edge_index', 'atom_type'):
                assert ident[k] == ref[k], f'{run}: graph mismatch on {k}'
        S = per_layer_scores(model, data, device)
        n = ident['num_nodes']
        for s in S:
            assert np.allclose(s.sum(axis=1), 1.0, atol=1e-4), 'column means must sum to 1'
        pl = peak_layer(run)
        s_peak = S[pl - 1]                                    # (H, n)
        s_mean = s_peak.mean(axis=0)
        j = int(s_mean.argmax())
        panels.append({
            'run': run, 'label': lab, 'attn_mode': config.get('model', {}).get('attn_mode', 'gps'),
            'peak_layer': pl,
            's_per_layer_head': [s.tolist() for s in S],      # [L][H][n]
            's_peak_headmean': s_mean.tolist(),
            'argmax_node': j,
            'argmax_s_headmean': float(s_mean[j]),
            'argmax_frac_heads_above_eps': float((s_peak[:, j] > EPS).mean()),
            # the paper's per-graph sink rate at this layer: max_j (1/H) sum_h 1[s_j^h > eps]
            'graph_sink_rate': float((s_peak > EPS).mean(axis=0).max()),
            'n_times_max_s': float(n * s_mean.max()),
        })
        print(f"{lab:>12s}: peak layer {pl}, argmax node {j}, s={s_mean[j]:.3f}, "
              f"n*s={n * s_mean[j]:.2f}, heads>eps at argmax {panels[-1]['argmax_frac_heads_above_eps']:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'graph': ref, 'eps': EPS, 'selection': args.select, 'n_atoms': args.n_atoms,
                   'panels': panels}, f)
    print(f'wrote {args.out} (test graph {ref["test_index"]}, n={ref["num_nodes"]})')


if __name__ == '__main__':
    main()
