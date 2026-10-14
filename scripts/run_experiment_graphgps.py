import argparse
import copy
import os
import sys

import numpy as np
import torch
import yaml
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphgps.model import InstrumentedGPS
from src.graphgps.datasets import get_dataloaders, DATASET_INFO
from src.graphgps.train import train_model, eval_epoch, build_criterion
from src.diagnostics_v2 import run_diagnostics_v2, aggregate_v2, compute_stability, save_suite


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path, base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.dirname(config_path), 'base.yaml')

    with open(base_path, 'r') as f:
        base = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        override = yaml.safe_load(f) or {}

    return deep_merge(base, override)


def apply_cli_overrides(config, overrides):
    for key, value in overrides.items():
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            d = d[part]
        old_val = d.get(parts[-1])
        if isinstance(old_val, bool):
            value = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_val, int):
            value = int(value)
        elif isinstance(old_val, float):
            value = float(value)
        d[parts[-1]] = value
    return config


def main():
    parser = argparse.ArgumentParser(description='Run a single GraphGPS experiment')
    parser.add_argument('--config', required=True, help='Path to experiment YAML config')
    parser.add_argument('--base-config', default=None, help='Path to base YAML config')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--device', default=None, help='Device (cuda, cpu, cuda:0, etc)')

    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = 'true'
                i += 1
        else:
            i += 1

    config = load_config(args.config, args.base_config)
    if overrides:
        config = apply_cli_overrides(config, overrides)

    seed = config.get('training', {}).get('seed')
    if seed is not None:
        from src.seed import set_seed
        set_seed(int(seed))
        sid = f"-s{int(seed)}"
        if not str(config['experiment_id']).endswith(sid):
            config['experiment_id'] = f"{config['experiment_id']}{sid}"

    experiment_id = config['experiment_id']
    dataset_name = config['data']['dataset']
    task = config['architecture']['task']
    use_wandb = not args.no_wandb

    print(f"=" * 60)
    print(f"Experiment: {experiment_id}")
    print(f"Dataset: {dataset_name}, Task: {task}")
    print(f"VNode: {config['vnode']['enabled']}, PE: {config['pe']['type']}, "
          f"Layers: {config['architecture']['num_layers']}, MPNN: {config['architecture']['mpnn']}")
    print(f"=" * 60)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = os.path.join(config['logging']['save_dir'], experiment_id)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=experiment_id,
            config=config,
            tags=[dataset_name, f"vnode={'on' if config['vnode']['enabled'] else 'off'}",
                    f"pe={config['pe']['type']}", f"layers={config['architecture']['num_layers']}"],
        )

    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    model = InstrumentedGPS(config, dataset_info).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    print("\nTraining...")
    results = train_model(model, train_loader, val_loader, config, device, task, save_dir=save_dir, use_wandb=use_wandb)
    print(f"\nBest val {results['metric_name']}: {results['best_val_metric']:.4f}")

    criterion = build_criterion(task)
    test_loss, test_metric, metric_name = eval_epoch(model, test_loader, criterion, device, task)
    print(f"Test {metric_name}: {test_metric:.4f}")

    if use_wandb:
        wandb.log({f'test/{metric_name}': test_metric, 'test/loss': test_loss})

    if config['diagnostics']['enabled']:
        # Quick end-of-training check with the diagnostics (the paper's numbers come from
        # scripts/run_eval_suite.py on the full test set, which writes eval_summary.json).
        print("\nRunning diagnostics...")
        results_v2 = run_diagnostics_v2(model, test_loader, device,
                                        max_graphs=config['diagnostics']['max_graphs'], spectra=False,
                                        has_vnode=config.get('vnode', {}).get('enabled', False))
        agg, layers = aggregate_v2(results_v2)
        save_suite(results_v2, agg, layers, compute_stability(results_v2), save_dir, suffix='_train')
        for key in ('max_sink_score', 'overall_sink_rate', 'norm_sink_score'):
            vals = [v for v in agg.get(key, {}).get('mean', []) if v == v]
            if vals:
                print(f"  {key}: peak={max(vals):.4f} final={agg[key]['mean'][-1]:.4f}")

        if use_wandb:
            for key in ['max_sink_score', 'overall_sink_rate', 'norm_sink_score']:
                vals = [v for v in agg.get(key, {}).get('mean', []) if v == v]
                if vals:
                    wandb.log({f'diag/{key}_max': float(max(vals)), f'diag/{key}_final': float(vals[-1])})

    if use_wandb:
        wandb.finish()
    print(f"\nExperiment {experiment_id} complete. Results saved to {save_dir}/")

if __name__ == '__main__':
    main()
