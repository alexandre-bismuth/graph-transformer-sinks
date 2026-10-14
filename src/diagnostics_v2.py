"""Diagnostics walker.

- Passes each graph's own edge_index to the Dirichlet energy.
- Supports full-test evaluation (max_graphs=None).
- Collects logits and value norms, virtual-node identity, attention spectra and
  cross-layer sink-identity stability.

Works with both InstrumentedGRIT (stashes attn_logits/value_norms during forward when
`collect_extra`) and InstrumentedGPS (exposes compute_logits_values(layer, x_graph)).
"""
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from .metrics_v2 import compute_all_metrics_v2


@torch.no_grad()
def run_diagnostics_v2(model, loader, device, max_graphs=None, spectra=True,
                       has_vnode=False):
    model.eval()
    model._register_attn_hooks()
    for layer in getattr(model, 'layers', []):
        mha = getattr(layer, 'mha', None)
        if mha is not None:
            mha.collect_extra = True

    num_layers = model.num_layers
    results = {l: [] for l in range(num_layers + 1)}
    processed = 0

    for batch in tqdm(loader, desc='Diagnostics'):
        if max_graphs is not None and processed >= max_graphs:
            break
        batch = batch.to(device)
        _ = model(batch, collect_diagnostics=True)

        bvec = model.layer_data[0]['batch']            # CPU tensor, node -> graph id
        counts = torch.bincount(bvec)
        offsets = counts.cumsum(0) - counts
        ei = batch.edge_index.cpu()
        e_gid = bvec[ei[0]]

        attn_logits = getattr(model, 'attn_logits', None)
        value_norms = getattr(model, 'value_norms', None)
        clamp_fracs = getattr(model, 'clamp_fracs', None)   # per layer (B,) or None
        can_offline = hasattr(model, 'compute_logits_values')

        for g in range(len(counts)):
            if max_graphs is not None and processed >= max_graphs:
                break
            n_g = int(counts[g].item())
            node_mask = (bvec == g)
            sel = (e_gid == g)
            local_ei = ei[:, sel] - offsets[g]
            vnode_idx = (n_g - 1) if has_vnode else None

            for l in range(num_layers + 1):
                Hg = model.layer_data[l]['h'][node_mask]
                attn_g = logits_g = vnorm_g = None
                if l > 0:
                    aw = model.attn_weights[l - 1]
                    if aw is not None and g < aw.size(0):
                        attn_g = aw[g, :, :n_g, :n_g].float()
                    if attn_logits is not None and l - 1 < len(attn_logits) \
                            and attn_logits[l - 1] is not None and g < attn_logits[l - 1].size(0):
                        logits_g = attn_logits[l - 1][g, :, :n_g, :n_g].float()
                    if value_norms is not None and l - 1 < len(value_norms) \
                            and value_norms[l - 1] is not None and g < value_norms[l - 1].size(0):
                        vnorm_g = value_norms[l - 1][g, :, :n_g].float()
                    if logits_g is None and attn_g is not None and can_offline:
                        x_in = model.layer_data[l - 1]['h'][node_mask]
                        try:
                            logits_g, vnorm_g = model.compute_logits_values(l - 1, x_in)
                        except Exception:
                            logits_g, vnorm_g = None, None

                m = compute_all_metrics_v2(Hg, local_ei, attn=attn_g, logits=logits_g,
                                           value_norms=vnorm_g, vnode_idx=vnode_idx,
                                           spectra=spectra)
                if l > 0 and clamp_fracs is not None and l - 1 < len(clamp_fracs) \
                        and clamp_fracs[l - 1] is not None:
                    for k_, v_ in clamp_fracs[l - 1].items():
                        if v_ is not None and g < len(v_):
                            m[k_] = float(v_[g])
                results[l].append(m)
            processed += 1

    model._remove_attn_hooks()
    for layer in getattr(model, 'layers', []):
        mha = getattr(layer, 'mha', None)
        if mha is not None:
            mha.collect_extra = False
    print(f"Processed {processed} graphs")
    return results


def compute_stability(results):
    """Cross-layer sink-identity consistency + mean top1/top2 dominance."""
    layers = sorted(k for k in results.keys() if k > 0)
    if not layers:
        return {}
    n_graphs = len(results[layers[0]])
    consist, dom = [], []
    for i in range(n_graphs):
        ids = [results[l][i].get('max_sink_node') for l in layers
               if i < len(results[l]) and results[l][i].get('max_sink_node') is not None]
        if len(ids) >= 2:
            same = sum(1 for a, b in zip(ids[:-1], ids[1:]) if a == b)
            consist.append(same / (len(ids) - 1))
        d = [results[l][i].get('top1_top2_ratio') for l in layers
             if i < len(results[l]) and results[l][i].get('top1_top2_ratio') is not None]
        d = [x for x in d if x is not None and np.isfinite(x)]
        if d:
            dom.append(float(np.mean(d)))
    return {
        'sink_identity_consistency_mean': float(np.mean(consist)) if consist else float('nan'),
        'top1_top2_dominance_mean': float(np.mean(dom)) if dom else float('nan'),
    }


def aggregate_v2(results):
    """Scalar aggregation: mean/std per layer for every scalar key."""
    layers = sorted(results.keys())
    keys = set()
    for l in layers:
        for m in results[l]:
            for k, v in m.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    keys.add(k)
    agg = {}
    for name in sorted(keys):
        means, stds, ns = [], [], []
        for l in layers:
            vals = [m[name] for m in results[l]
                    if name in m and m[name] is not None and np.isfinite(m[name])]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                ns.append(len(vals))
            else:
                means.append(float('nan'))
                stds.append(float('nan'))
                ns.append(0)
        agg[name] = {'mean': means, 'std': stds, 'n': ns}
    return agg, layers


def save_suite(results, agg, layers, stability, out_dir, suffix=''):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'eval_suite{suffix}.pkl'), 'wb') as f:
        pickle.dump({'results': results, 'agg': agg, 'layers': layers,
                     'stability': stability}, f)
    summary = {'layers': layers, 'stability': stability,
               'agg': {k: v for k, v in agg.items()}}
    with open(os.path.join(out_dir, f'eval_summary{suffix}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"Saved eval_suite{suffix}.pkl / eval_summary{suffix}.json to {out_dir}")
