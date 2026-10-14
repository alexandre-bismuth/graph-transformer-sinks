"""Diagnostic metrics.

- Dirichlet energy on each graph's own edge_index, with normalised variants (per-edge,
  Rayleigh).
- Node-norm ratio, squared (`max_to_mean_ratio`) and unsquared.
- eps-grid sink rates under three estimators:
    sink_rate_eps*   the paper's estimator: max over nodes of the fraction of heads
                     whose received attention exceeds eps (`overall_sink_rate`)
    headrate_eps*    per-layer literal form: fraction of heads in which >=1 node exceeds eps
    norm_sink_score  n-normalised max sink score (multiples of uniform attention)
- attention-operator spectra: SLEM of head-averaged A and of the lazy kernel (A+I)/2;
  effective rank; leave-sink-out SLEM. The informative quantity is the random-init value
  (run with --randinit): comparing the trained SLEM with the sink rate is close to circular.
- logit-growth statistics: log-log slope of mean |received logit| vs key-node
  hidden norm, and Spearman(received attention, node norm).
- value-norm statistics (no-op vs broadcast): sink-node value norm relative to
  the graph mean.
- massive-activation criterion: max|h_ij| / median|h_ij|.
"""
import numpy as np
import torch

EPS_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _spearman(a, b):
    """Spearman rho of two 1-D tensors (ties handled by average ranks via argsort twice)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def compute_sink_metrics(attn, eps_grid=EPS_GRID):
    """attn: (H, N, N), rows (dim 1) = queries, row-stochastic over keys (dim 2)."""
    H, N, _ = attn.shape
    per_head_recv = attn.mean(dim=1)            # (H, N) attention received per node/head
    node_score = per_head_recv.mean(dim=0)      # (N,) head-averaged sink score
    out = {
        'max_sink_score': node_score.max().item(),
        'max_sink_node': int(node_score.argmax().item()),
        'norm_sink_score': node_score.max().item() * N,
        'sink_score_per_node': node_score,      # tensor; excluded from scalar aggregation
        'per_head_recv': per_head_recv,         # tensor; for downstream analyses
    }
    if N >= 2:
        top2 = torch.topk(node_score, k=2).values
        out['top1_top2_ratio'] = (top2[0] / (top2[1] + 1e-12)).item()
    else:
        out['top1_top2_ratio'] = float('nan')
    for eps in eps_grid:
        exceed = per_head_recv > eps
        out[f'sink_rate_eps{eps}'] = exceed.float().mean(dim=0).max().item()
        out[f'headrate_eps{eps}'] = exceed.any(dim=1).float().mean().item()
    out['overall_sink_rate'] = out['sink_rate_eps0.3']   # headline estimator
    return out


def _slem(M):
    """Second-largest eigenvalue modulus of a (near-)stochastic matrix."""
    try:
        ev = torch.linalg.eigvals(M.float())
        mags = ev.abs().sort(descending=True).values
        return mags[1].item() if mags.numel() > 1 else float('nan')
    except Exception:
        return float('nan')


def compute_attention_spectra(attn, sink_node=None):
    """Spectra of the head-averaged attention operator."""
    A = attn.mean(dim=0)
    N = A.shape[0]
    if N < 3:
        return {'slem_A': float('nan'), 'slem_lazy': float('nan'),
                'attn_effective_rank': float('nan'), 'slem_lazy_nosink': float('nan')}
    I = torch.eye(N, dtype=A.dtype)
    out = {
        'slem_A': _slem(A),
        'slem_lazy': _slem(0.5 * (A + I)),
    }
    try:
        S = torch.linalg.svdvals(A.float())
        p = S / (S.sum() + 1e-12)
        p = p[p > 1e-12]
        out['attn_effective_rank'] = float(torch.exp(-(p * torch.log(p)).sum()).item())
    except Exception:
        out['attn_effective_rank'] = float('nan')
    out['slem_lazy_nosink'] = float('nan')
    if sink_node is not None and N > 3:
        keep = torch.tensor([i for i in range(N) if i != sink_node])
        B = A[keep][:, keep]
        rs = B.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out['slem_lazy_nosink'] = _slem(0.5 * (B / rs + torch.eye(N - 1, dtype=A.dtype)))
    return out


def compute_dirichlet_v2(Hf, edge_index_local):
    """Dirichlet energy on each graph's own edges."""
    E = edge_index_local.shape[1]
    if E == 0:
        return {'dirichlet_raw': 0.0, 'dirichlet_per_edge': 0.0, 'dirichlet_rayleigh': 0.0}
    src, dst = edge_index_local[0], edge_index_local[1]
    d = (Hf[src] - Hf[dst]).pow(2).sum().item()
    fro = Hf.pow(2).sum().item()
    return {
        'dirichlet_raw': d,
        'dirichlet_per_edge': d / E,
        'dirichlet_rayleigh': d / (fro + 1e-12),
    }


def compute_norm_stats_v2(Hf):
    norms = Hf.norm(dim=-1)
    mx = norms.max().item()
    mn = norms.mean().item()
    med = norms.median().item()
    abs_h = Hf.abs().flatten()
    med_abs = abs_h.median().item()
    return {
        'max_to_mean_ratio': (mx ** 2) / (mn ** 2 + 1e-12),   # squared ratio
        'norm_ratio_unsq': mx / (mn + 1e-12),
        'norm_ratio_median': mx / (med + 1e-12),
        'max_abs_over_median_abs': abs_h.max().item() / (med_abs + 1e-12),
        'max_norm': mx,
        'mean_norm': mn,
        'max_norm_node': int(norms.argmax().item()),
        'norm_std': norms.std().item() if norms.numel() > 1 else 0.0,
    }


def compute_matrix_entropy(Hf):
    try:
        S = torch.linalg.svdvals(Hf.float())
    except Exception:
        return 0.0
    S_sq = S.pow(2)
    total = S_sq.sum()
    if total < 1e-10:
        return 0.0
    p = S_sq / total
    p = p[p > 1e-10]
    entropy = -(p * torch.log(p)).sum().item()
    max_entropy = np.log(len(S))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_anisotropy(Hf):
    try:
        S = torch.linalg.svdvals(Hf.float())
    except Exception:
        return 1.0
    S_sq = S.pow(2)
    total = S_sq.sum()
    if total < 1e-10:
        return 1.0
    return (S_sq[0] / total).item()


def compute_logit_growth(node_norms, node_score, logits=None):
    """How do received logits/attention scale with key-node hidden norm?"""
    out = {'attn_norm_spearman': _spearman(node_score.numpy(), node_norms.numpy())}
    if logits is not None and torch.isfinite(logits).all():
        recv = logits.abs().mean(dim=(0, 1))               # (N,) mean |logit| per key node
        x = torch.log(node_norms.clamp_min(1e-8))
        y = torch.log(recv.clamp_min(1e-8))
        if x.numel() >= 5 and x.std() > 1e-8:
            xc, yc = x - x.mean(), y - y.mean()
            out['logit_norm_slope'] = float(((xc * yc).sum() / (xc.pow(2).sum() + 1e-12)).item())
        else:
            out['logit_norm_slope'] = float('nan')
        out['logit_norm_spearman'] = _spearman(recv.numpy(), node_norms.numpy())
    else:
        out['logit_norm_slope'] = float('nan')
        out['logit_norm_spearman'] = float('nan')
    return out


def compute_value_stats(value_norms, sink_node):
    """No-op (sink value ~ 0) vs broadcast (sink value informative)."""
    v = value_norms.mean(dim=0) if value_norms.dim() == 2 else value_norms
    m = v.mean() + 1e-12
    return {
        'sink_value_ratio': (v[sink_node] / m).item(),
        'min_value_ratio': (v.min() / m).item(),
    }


def compute_consensus_residual(Hf):
    """Token-uniformity residual ||X - 1 xbar^T||_F / ||X||_F (Dong et al., 2021): 0 when every
    node carries the same representation (complete over-mixing), 1 when the mean is zero."""
    tot = Hf.pow(2).sum().sqrt().item()
    if tot < 1e-12:
        return float('nan')
    return (Hf - Hf.mean(dim=0, keepdim=True)).pow(2).sum().sqrt().item() / tot


def compute_all_metrics_v2(H_graph, edge_index_local, attn=None, logits=None,
                           value_norms=None, vnode_idx=None, spectra=True):
    Hf = H_graph.float()
    result = {
        'matrix_entropy': compute_matrix_entropy(Hf),
        'anisotropy': compute_anisotropy(Hf),
        'consensus_residual': compute_consensus_residual(Hf),
        'num_nodes': int(Hf.shape[0]),
    }
    result.update(compute_norm_stats_v2(Hf))
    result.update(compute_dirichlet_v2(Hf, edge_index_local))

    if attn is not None:
        sink = compute_sink_metrics(attn)
        node_score = sink.pop('sink_score_per_node')
        sink.pop('per_head_recv')
        result.update(sink)
        norms = Hf.norm(dim=-1)
        result.update(compute_logit_growth(norms, node_score, logits=logits))
        if spectra:
            result.update(compute_attention_spectra(attn, sink_node=result['max_sink_node']))
        if value_norms is not None:
            result.update(compute_value_stats(value_norms, result['max_sink_node']))
        if vnode_idx is not None:
            result['vnode_is_sink'] = float(result['max_sink_node'] == vnode_idx)
            result['vnode_recv_score'] = node_score[vnode_idx].item()
    return result
