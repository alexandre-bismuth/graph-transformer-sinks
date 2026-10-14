"""Symmetry--locality sink-barrier certificate.

For each test graph, layer, head, computes:
  s_j        = per-head received attention (column mean)  -- the paper's sink score
  ell_u^(r)  = row-u attention mass at graph distance > r (realized leakage)
  eps_r      = max_u ell_u^(r)                            (classic leakage)
  T_j^(r)    = (|B(j,r)| + sum_{u: d(u,j)>r} ell_u^(r)) / n   (tight locality ceiling)
  orbit_j    = size of j's orbit under label-preserving automorphisms (capped VF2;
               a partial automorphism set only shrinks orbits, so 1/|orbit| stays a
               valid (looser) upper bound)
  U_j        = min(1/orbit_j, min_r T_j^(r))              (certified ceiling)
  kappa      = Dobrushin coefficient 0.5 * max_{u,v} ||A_u - A_v||_1
  gamma_u(r) = min in-ball total logit - max out-of-ball total logit (margin)

Validity: s_j <= T_j^(r) holds for any row-stochastic A (mass counting, no
equivariance needed); s_j <= 1/orbit_j additionally relies on the model's
permutation equivariance, so violations are counted and reported (should be 0
up to float tolerance).

Outputs <run_dir>/theory_certificate.json (aggregates) and .pkl (per-graph arrays).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_eval_suite import load_run_config, detect_arch, build_everything

RADII = (0, 1, 2, 3)


def bfs_distances(n, edge_index):
    """All-pairs unweighted distances via scipy csgraph (undirected)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    ei = edge_index.numpy()
    w = np.ones(ei.shape[1])
    adj = coo_matrix((w, (ei[0], ei[1])), shape=(n, n))
    d = shortest_path(adj, method='BF' if n < 4 else 'D', directed=False, unweighted=True)
    return d  # inf for disconnected pairs


def automorphism_orbits(n, edge_index, node_labels, edge_labels, cap=2000):
    """Orbit sizes under label-preserving automorphisms (VF2, enumeration capped)."""
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match, categorical_edge_match
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, lab=tuple(np.atleast_1d(node_labels[i]).tolist()))
    ei = edge_index.numpy()
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a == b:
            continue
        lab = tuple(np.atleast_1d(edge_labels[k]).tolist()) if edge_labels is not None else (0,)
        G.add_edge(a, b, lab=lab)
    gm = GraphMatcher(G, G, node_match=categorical_node_match('lab', None),
                      edge_match=categorical_edge_match('lab', None))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    count = 0
    for mapping in gm.isomorphisms_iter():
        for a, b in mapping.items():
            union(a, b)
        count += 1
        if count >= cap:
            break
    sizes = {}
    for i in range(n):
        sizes.setdefault(find(i), 0)
        sizes[find(i)] += 1
    orbit_size = np.array([sizes[find(i)] for i in range(n)], dtype=np.int64)
    roots = np.array([find(i) for i in range(n)])
    return orbit_size, roots, count


def per_graph_certificate(attn, logits, dist, orbit_size, theta):
    """attn: (H, n, n) row-stochastic; dist: (n, n); returns per-head dicts."""
    H, n, _ = attn.shape
    A = attn.numpy()
    out = []
    orbit_ceiling = 1.0 / orbit_size  # (n,)
    for h in range(H):
        Ah = A[h]
        s = Ah.mean(axis=0)  # column means = s_j
        T = np.full((len(RADII), n), np.inf)
        eps = np.zeros(len(RADII))
        for ri, r in enumerate(RADII):
            far = dist > r  # (n, n): far[u, v] = d(u,v) > r
            ell = (Ah * far).sum(axis=1)  # row leakage at radius r
            eps[ri] = ell.max()
            ball = (~far).sum(axis=0)     # |B(j,r)| (dist symmetric)
            # tight ceiling: queries outside B(j,r) contribute at most their leakage
            far_q = dist.T > r            # far_q[j, u] = d(u,j) > r
            T[ri] = (ball + far_q @ ell) / n
        T_best = T.min(axis=0)
        U = np.minimum(orbit_ceiling, T_best)
        viol_loc = int((s > T_best + 1e-4).sum())
        viol_orb = int((s > orbit_ceiling + 1e-4).sum())
        # Dobrushin
        At = torch.from_numpy(Ah)
        kappa = 0.5 * torch.cdist(At, At, p=1).max().item()
        head = {
            's_max': float(s.max()), 'argmax_is_sink': bool(s.max() > theta),
            'U_at_argmax': float(U[s.argmax()]), 'U_max': float(U.max()),
            'certified_no_sink': bool(U.max() < theta),
            'slack_at_argmax': float(U[s.argmax()] - s.max()),
            'eps_r': eps.tolist(), 'kappa': float(kappa),
            'viol_locality': viol_loc, 'viol_orbit': viol_orb,
            'orbit_size_argmax': int(orbit_size[s.argmax()]),
        }
        if logits is not None:
            Lh = logits[h].numpy()
            margins = []
            for ri, r in enumerate(RADII):
                far = dist > r
                m_r = []
                for u in range(n):
                    fu = far[u]
                    if fu.any() and (~fu).any():
                        m_r.append(float(Lh[u][~fu].min() - Lh[u][fu].max()))
                margins.append(float(np.mean(m_r)) if m_r else None)
            head['logit_margin_r'] = margins
        out.append(head)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--max-graphs', type=int, default=300)
    p.add_argument('--theta', type=float, default=0.3)
    p.add_argument('--orbit-cap-nodes', type=int, default=80)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_run_config(args.run_dir)
    arch = detect_arch(config)
    has_vnode = bool(config['data'].get('virtual_node', False))

    model, test_loader, _, _ = build_everything(config, arch, device)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model._register_attn_hooks()
    for layer in getattr(model, 'layers', []):
        mha = getattr(layer, 'mha', None)
        if mha is not None:
            mha.collect_extra = True

    per_graph = []
    processed = 0
    with torch.no_grad():
        for batch in test_loader:
            if processed >= args.max_graphs:
                break
            batch = batch.to(device)
            _ = model(batch, collect_diagnostics=True)
            bvec = model.layer_data[0]['batch']
            counts = torch.bincount(bvec)
            offsets = counts.cumsum(0) - counts
            ei = batch.edge_index.cpu()
            e_gid = bvec[ei[0]]
            attn_logits = getattr(model, 'attn_logits', None)
            x_cpu = batch.x.cpu()
            ea_cpu = batch.edge_attr.cpu() if getattr(batch, 'edge_attr', None) is not None else None

            for g in range(len(counts)):
                if processed >= args.max_graphs:
                    break
                n_g = int(counts[g].item())
                if n_g < 3:
                    continue
                node_mask = (bvec == g)
                sel = (e_gid == g)
                local_ei = ei[:, sel] - offsets[g]
                dist = bfs_distances(n_g, local_ei)
                dist[np.isinf(dist)] = n_g + 100  # disconnected = beyond any radius

                labels = x_cpu[node_mask].numpy()
                elabels = ea_cpu[sel].numpy() if ea_cpu is not None else None
                if n_g <= args.orbit_cap_nodes:
                    orbit_size, roots, n_auts = automorphism_orbits(
                        n_g, local_ei, labels, elabels)
                else:
                    orbit_size = np.ones(n_g, dtype=np.int64)  # trivial (valid) fallback
                    roots, n_auts = np.arange(n_g), 0

                graph_entry = {'n': n_g, 'layers': {},
                               'orbit_sizes': orbit_size.tolist(),
                               'n_automorphisms_seen': int(n_auts)}
                for l in range(1, model.num_layers + 1):
                    aw = model.attn_weights[l - 1]
                    if aw is None or g >= aw.size(0):
                        continue
                    attn_g = aw[g, :, :n_g, :n_g].float().cpu()
                    logits_g = None
                    if attn_logits is not None and l - 1 < len(attn_logits) \
                            and attn_logits[l - 1] is not None and g < attn_logits[l - 1].size(0):
                        logits_g = attn_logits[l - 1][g, :, :n_g, :n_g].float().cpu()
                    # equivariance sanity: within-orbit spread of s per head
                    heads = per_graph_certificate(attn_g, logits_g, dist,
                                                  orbit_size, args.theta)
                    s_all = attn_g.mean(dim=1).numpy()  # (H, n) column means
                    spread = 0.0
                    for root in np.unique(roots):
                        idx = np.where(roots == root)[0]
                        if len(idx) > 1:
                            spread = max(spread, float(np.ptp(s_all[:, idx], axis=1).max()))
                    graph_entry['layers'][l] = {'heads': heads,
                                                'orbit_spread': spread}
                per_graph.append(graph_entry)
                processed += 1

    # ---------- aggregate ----------
    all_heads = [(e['n'], l, hd) for e in per_graph for l, ld in e['layers'].items()
                 for hd in ld['heads']]
    total = len(all_heads)
    certified = sum(1 for _, _, h in all_heads if h['certified_no_sink'])
    sinky = sum(1 for _, _, h in all_heads if h['argmax_is_sink'])
    viol = sum(h['viol_locality'] for _, _, h in all_heads)
    viol_orb = sum(h['viol_orbit'] for _, _, h in all_heads)
    eps_by_r = np.array([h['eps_r'] for _, _, h in all_heads])
    slack = np.array([h['slack_at_argmax'] for _, _, h in all_heads])
    kappa = np.array([h['kappa'] for _, _, h in all_heads])
    smax = np.array([h['s_max'] for _, _, h in all_heads])
    orb_argmax = np.array([h['orbit_size_argmax'] for _, _, h in all_heads])
    spread = max((ld['orbit_spread'] for e in per_graph for ld in e['layers'].values()),
                 default=0.0)
    ns = np.array([n for n, _, _ in all_heads])

    def by_layer(fn):
        out = {}
        for _, l, h in all_heads:
            out.setdefault(l, []).append(fn(h))
        return {l: float(np.mean(v)) for l, v in sorted(out.items())}

    summary = {
        'run_dir': args.run_dir, 'theta': args.theta, 'n_graphs': processed,
        'n_head_instances': total,
        'certified_no_sink_frac': certified / max(total, 1),
        'observed_sink_frac': sinky / max(total, 1),
        'locality_bound_violations': int(viol),
        'orbit_bound_violations': int(viol_orb),
        'max_within_orbit_spread': spread,
        'eps_r_mean': eps_by_r.mean(axis=0).tolist(),
        'eps_r_median': np.median(eps_by_r, axis=0).tolist(),
        'slack_at_argmax_mean': float(slack.mean()),
        'slack_at_argmax_median': float(np.median(slack)),
        'kappa_mean': float(kappa.mean()),
        'kappa_vs_smax_corr': float(np.corrcoef(kappa, smax)[0, 1]) if total > 2 else None,
        'orbit_size_at_argmax_mean': float(orb_argmax.mean()),
        'sink_heads_with_orbit_gt3': int(sum(1 for _, _, h in all_heads
                                             if h['argmax_is_sink'] and h['orbit_size_argmax'] > 3)),
        'certified_frac_by_layer': by_layer(lambda h: h['certified_no_sink']),
        'smax_mean_by_layer': by_layer(lambda h: h['s_max']),
        'n_quartile_certified': {},
    }
    if processed > 8:
        qs = np.quantile(ns, [0.25, 0.5, 0.75])
        bins = np.digitize(ns, qs)
        for b in range(4):
            m = bins == b
            if m.any():
                summary['n_quartile_certified'][f'q{b + 1}'] = {
                    'certified_frac': float(np.mean([h['certified_no_sink']
                                                     for (_, _, h), mm in zip(all_heads, m) if mm])),
                    'eps1_mean': float(eps_by_r[m, 1].mean()),
                    'n_mean': float(ns[m].mean())}

    out_json = os.path.join(args.run_dir, 'theory_certificate.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.run_dir, 'theory_certificate.pkl'), 'wb') as f:
        pickle.dump(per_graph, f)
    print(json.dumps(summary, indent=1))
    print(f"saved {out_json}")


if __name__ == '__main__':
    main()
