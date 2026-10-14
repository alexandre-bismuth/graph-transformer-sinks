"""Evaluation suite: rebuild a trained run from its saved config and checkpoint and
recompute all diagnostics on the full test set: task metrics, sink rates on an eps grid
(three estimators), per-graph Dirichlet energy, virtual-node sink identity, cross-layer
sink stability, attention spectra, logit growth and value norms. --randinit evaluates
the untrained model as a control.

Usage:
  python scripts/run_eval_suite.py --run-dir outputs/zinc-grit
  python scripts/run_eval_suite.py --run-dir outputs/zinc-grit --randinit --seed 0
  python scripts/run_eval_suite.py --run-dir outputs/peptides-grit-Q4 --max-graphs 200
  # evaluate a trained GRIT checkpoint under a different scoring path
  # (no retraining; parameters are shared across the GRIT family). Writes
  # eval_suite_<suffix>.pkl / eval_summary_<suffix>.json / metrics_<suffix>.json only.
  python scripts/run_eval_suite.py --run-dir outputs/zinc-grit-s0 --attn-mode grit_noclamp
  # (off-distribution: a clamp-trained checkpoint is broken under other scoring paths)
  # re-evaluate an existing run without overwriting its result files:
  python scripts/run_eval_suite.py --run-dir outputs/zinc-grit-s0 --out-tag cr --no-spectra
"""
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.seed import set_seed
from src.diagnostics_v2 import run_diagnostics_v2, aggregate_v2, compute_stability, save_suite


def load_run_config(run_dir):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_fresh_config(config_path):
    """Merge an experiment YAML with configs/base.yaml (runner-style) — for --config
    mode, used to measure random-init diagnostics of arms without a run directory."""
    import copy

    def deep_merge(base, override):
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    base_path = os.path.join(os.path.dirname(config_path), 'base.yaml')
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    return deep_merge(base, override)


def detect_arch(config):
    return 'gps' if 'mpnn' in config.get('architecture', {}) else 'grit'


def build_everything(config, arch, device):
    if arch == 'gps':
        from src.graphgps.model import InstrumentedGPS as ModelCls
        from src.graphgps.datasets import get_dataloaders
        from src.graphgps.train import eval_epoch, build_criterion
    else:
        from src.model import InstrumentedGRIT as ModelCls
        from src.datasets import get_dataloaders
        from src.train import eval_epoch, build_criterion
    train_loader, val_loader, test_loader, info = get_dataloaders(config)
    model = ModelCls(config, info).to(device)
    return model, test_loader, eval_epoch, build_criterion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=None)
    p.add_argument('--config', default=None,
                   help='experiment YAML (randinit-only mode, no trained run needed)')
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--max-graphs', type=int, default=None,
                   help='default: full test set')
    p.add_argument('--randinit', action='store_true', help='untrained control')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-spectra', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--attn-mode', default=None,
                   help='override model.attn_mode at evaluation time (GRIT family only)')
    p.add_argument('--out-tag', default=None,
                   help='append _<tag> to every output filename (re-evaluating an existing '
                        'run without overwriting its result files)')
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.run_dir is not None:
        config = load_run_config(args.run_dir)
    else:
        assert args.config and args.randinit, '--config mode requires --randinit'
        config = load_fresh_config(args.config)
        tag = os.path.splitext(os.path.basename(args.config))[0]
        args.run_dir = os.path.join('outputs', f'randinit-{tag}')
        os.makedirs(args.run_dir, exist_ok=True)
    if args.attn_mode is not None:
        from src.model import GritMultiHeadAttention
        fam = GritMultiHeadAttention.GRIT_FAMILY
        assert detect_arch(config) == 'grit' and \
            config.get('model', {}).get('attn_mode', 'grit') in fam and args.attn_mode in fam, \
            '--attn-mode needs a GRIT-family run and a GRIT-family mode'
        config.setdefault('model', {})['attn_mode'] = args.attn_mode
    arch = detect_arch(config)
    has_vnode = config.get('vnode', {}).get('enabled', False)
    task = config['architecture']['task']
    print(f"run={args.run_dir} arch={arch} task={task} vnode={has_vnode} "
          f"randinit={args.randinit} device={device}")

    set_seed(args.seed)
    model, test_loader, eval_epoch, build_criterion = build_everything(config, arch, device)

    suffix = ''
    if args.randinit:
        suffix = f'_randinit_s{args.seed}'
    else:
        ckpt_path = os.path.join(args.run_dir, args.ckpt)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"loaded {ckpt_path}")
    if args.attn_mode is not None:
        suffix += f'_eval{args.attn_mode}'
    if args.out_tag:
        suffix += f'_{args.out_tag}'

    # task metrics
    if not args.randinit:
        criterion = build_criterion(task)
        test_loss, test_metric, metric_name = eval_epoch(model, test_loader, criterion, device, task)
        n_params = sum(p_.numel() for p_ in model.parameters())
        metrics = {'test_loss': float(test_loss), f'test_{metric_name}': float(test_metric),
                   'metric_name': metric_name, 'n_params': int(n_params),
                   'ckpt': args.ckpt}
        if args.attn_mode is not None:
            metrics['eval_attn_mode'] = args.attn_mode
        with open(os.path.join(args.run_dir, f'metrics{suffix}.json'), 'w') as f:
            json.dump(metrics, f, indent=1)
        print(f"task metrics: {metrics}")

    # diagnostics
    results = run_diagnostics_v2(model, test_loader, device,
                                 max_graphs=args.max_graphs,
                                 spectra=not args.no_spectra,
                                 has_vnode=has_vnode)
    agg, layers = aggregate_v2(results)
    stability = compute_stability(results)
    save_suite(results, agg, layers, stability, args.run_dir, suffix=suffix)

    # quick console summary
    for key in ('max_sink_score', 'overall_sink_rate', 'norm_sink_score',
                'slem_lazy', 'dirichlet_per_edge', 'logit_norm_slope',
                'attn_norm_spearman', 'sink_value_ratio', 'vnode_is_sink',
                'max_to_mean_ratio', 'norm_ratio_unsq', 'max_abs_over_median_abs'):
        if key in agg:
            vals = [v for v in agg[key]['mean'] if v == v]  # drop NaN
            if vals:
                print(f"  {key}: peak={max(vals):.4f} final={agg[key]['mean'][-1]:.4f}")
    print(f"  stability: {stability}")


if __name__ == '__main__':
    main()
