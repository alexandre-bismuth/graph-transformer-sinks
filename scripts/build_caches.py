"""Build the positional-encoding caches (data/pe_cache/*.pt) that the given configs need,
so training and evaluation only ever load them. CPU-heavy: the Peptides RRWP-17 cache is
dense n^2 per graph, ~25 GB. Configs sharing a cache file are
built once; already-present files are skipped (the loaders check the exact path).

Usage:
  python scripts/build_caches.py configs/zinc_grit.yaml configs/zinc_graphgps.yaml \
      configs/peptides_grit.yaml
Prints one line per (dataset, pe) with the cache paths and their sizes.
  python scripts/build_caches.py --print-paths <configs>   # list the files, build nothing
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_eval_suite import load_fresh_config, detect_arch


def cache_paths(config):
    root = config['data'].get('root', 'data')
    name, pe, dim = config['data']['dataset'], config['pe']['type'], config['pe']['dim']
    v = config.get('vnode', {})
    tag = '_vnodepe' if v.get('enabled', False) and v.get('pe_mode', 'padded') == 'recomputed' else ''
    if pe in (None, 'none'):
        return []
    return [os.path.join(root, 'pe_cache', f'{name}_{s}_{pe}{dim}{tag}.pt')
            for s in ('train', 'val', 'test')]


def load_any(path):
    """An experiment YAML (merged with its base.yaml) or a run dir's saved config.yaml (already
    merged: no base.yaml next to it)."""
    if os.path.exists(os.path.join(os.path.dirname(path), 'base.yaml')):
        return load_fresh_config(path)
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--print-paths', action='store_true',
                    help='only print the cache files these configs read (one per line)')
    args = ap.parse_args()
    if args.print_paths:
        seen = []
        for c in args.configs:
            for q in cache_paths(load_any(c)):
                if q not in seen:
                    seen.append(q)
                    print(q)
        return
    done = set()
    for c in args.configs:
        config = load_any(c)
        paths = tuple(cache_paths(config))
        if not paths or paths in done:
            continue
        done.add(paths)
        if all(os.path.exists(p) for p in paths):
            print(f'[cache] present: {paths[0]} (+val/test)')
            continue
        t0 = time.time()
        if detect_arch(config) == 'gps':
            from src.graphgps.datasets import get_datasets
        else:
            from src.datasets import get_datasets
        get_datasets(config)
        sizes = ', '.join(f'{os.path.basename(p)} {os.path.getsize(p) / 1e9:.2f} GB' for p in paths)
        print(f'[cache] built in {time.time() - t0:.0f}s: {sizes}')


if __name__ == '__main__':
    main()
