"""Collect every finished run into per-arm aggregates with 95% CIs.

One command turns outputs/ into the numbers the paper quotes:

    python scripts/aggregate_results.py                 # markdown to stdout
    python scripts/aggregate_results.py -o readout.md --json readout.json

An "arm" is a run dir with its trailing -s<seed> stripped, so zinc-grit-dot-s0/1/2
aggregate into zinc-grit-dot with n=3.

Headline sink estimator (full test set) is
    nanmax over layers of eval_summary.json agg.overall_sink_rate.mean
which is metrics_v2's sink_rate_eps0.3: max over nodes of the fraction of heads whose
received attention exceeds 0.3.  Note that the neighbouring key `headrate_eps0.3` is a
different statistic and does not reproduce the quoted numbers.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# 95% two-sided t quantiles by degrees of freedom; >10 falls back to the normal value.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _num(x):
    """None for missing/NaN, float otherwise."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def nanmax(seq):
    vals = [v for v in (_num(x) for x in (seq or [])) if v is not None]
    return max(vals) if vals else None


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def collect_run(run_dir):
    """Every scalar we care about from one run dir. Missing pieces stay None."""
    # training snapshots ckpt.pt every 50 epochs and deletes it when the run finishes
    r = {'run': os.path.basename(run_dir),
         'trained': not os.path.exists(os.path.join(run_dir, 'ckpt.pt'))}

    m = load(os.path.join(run_dir, 'metrics.json'))
    if m:
        r['metric_name'] = m.get('metric_name')
        r['score'] = _num(m.get('test_mae') if m.get('metric_name') == 'mae' else m.get('test_ap'))
        r['n_params'] = m.get('n_params')

    e = load(os.path.join(run_dir, 'eval_summary.json'))
    if e:
        agg = e.get('agg', {})
        r['sink'] = nanmax(agg.get('overall_sink_rate', {}).get('mean'))
        r['max_sink_score'] = nanmax(agg.get('max_sink_score', {}).get('mean'))
        r['dirichlet'] = nanmax(agg.get('dirichlet_per_edge', {}).get('mean'))
        st = e.get('stability', {})
        r['sink_consistency'] = _num(st.get('sink_identity_consistency_mean'))
        r['dominance'] = _num(st.get('top1_top2_dominance_mean'))

    a = load(os.path.join(run_dir, 'sink_ablation_s0.json'))
    if a:
        c = a.get('conditions', {})
        for cond in ('none', 'sink', 'random', 'topdeg', 'sink2'):
            r['abl_' + cond] = _num(c.get(cond, {}).get('mae'))
        if r.get('abl_none'):
            r['abl_sink_ratio'] = r['abl_sink'] / r['abl_none'] if r.get('abl_sink') else None

    p = load(os.path.join(run_dir, 'perturbation_s0.json'))
    if p:
        b = p.get('buckets', {})
        # final layer = highest integer key present
        keys = sorted((int(k) for k in b), reverse=True)
        if keys:
            fl = b[str(keys[0])]
            for d in ('d0', 'd1', 'd2', 'd3plus'):
                r['pert_' + d] = _num(fl.get(d, {}).get('mean'))
            if r.get('pert_d0'):
                r['pert_reach'] = (r['pert_d3plus'] / r['pert_d0']) if r.get('pert_d3plus') else None

    c = load(os.path.join(run_dir, 'theory_certificate.json'))
    if c:
        r['cert_certified'] = _num(c.get('certified_no_sink_frac'))
        r['cert_observed'] = _num(c.get('observed_sink_frac'))
        r['cert_loc_viol'] = c.get('locality_bound_violations')
        r['cert_orbit_viol'] = c.get('orbit_bound_violations')
        r['cert_orbit_spread'] = _num(c.get('max_within_orbit_spread'))
        eps = c.get('eps_r_median') or []
        r['cert_eps1'] = _num(eps[0]) if eps else None

    s = load(os.path.join(run_dir, 'symmetry_probe.json'))
    if s:
        r['probe_pe'] = s.get('pe_type')
        devs = [_num(v.get('max_abs_deviation')) for v in s.get('sizes', {}).values()]
        devs = [d for d in devs if d is not None]
        r['probe_max_dev'] = max(devs) if devs else None
        ms = [_num(v.get('max_s')) for v in s.get('sizes', {}).values()]
        r['probe_max_s'] = max([v for v in ms if v is not None] or [None]) if ms else None
        vs = [_num(v.get('s_vnode')) for v in s.get('sizes', {}).values()]
        vs = [v for v in vs if v is not None]
        r['probe_s_vnode'] = max(vs) if vs else None
    return r


def stats(vals):
    """mean, sd, half-width of the 95% t interval, n."""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n == 1:
        return {'mean': mean, 'sd': None, 'ci': None, 'n': 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    t = T95.get(n - 1, 1.96)
    return {'mean': mean, 'sd': sd, 'ci': t * sd / math.sqrt(n), 'n': n}


def fmt(s, prec=4):
    if s is None:
        return '--'
    if s['n'] == 1:
        return f"{s['mean']:.{prec}f} (n=1)"
    return f"{s['mean']:.{prec}f} ± {s['ci']:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('-o', '--out', help='write markdown here (default: stdout only)')
    ap.add_argument('--json', help='also dump the raw aggregate as JSON')
    ap.add_argument('--include-partial', action='store_true',
                    help='do not flag runs that still have a training snapshot ckpt.pt (interrupted)')
    ap.add_argument('--seeded-only', action='store_true',
                    help='keep only seed-controlled runs (<arm>-s<k>); drops unsuffixed, unseeded '
                         'run directories')
    args = ap.parse_args()

    runs = []
    for d in sorted(glob.glob(os.path.join(args.outputs, '*'))):
        if not os.path.isdir(d):
            continue
        if args.seeded_only and not re.search(r'-s\d+$', os.path.basename(d)):
            continue
        r = collect_run(d)
        if r.get('score') is None and r.get('sink') is None:
            continue                      # nothing scored yet: not a result
        if not r['trained'] and not args.include_partial:
            r['partial'] = True
        runs.append(r)

    arms = defaultdict(list)
    for r in runs:
        arms[re.sub(r'-s\d+$', '', r['run'])].append(r)

    L = []
    L.append(f"# Readout — {len(runs)} scored runs across {len(arms)} arms\n")
    L.append("Sink = max-over-layers of the full-test `overall_sink_rate` "
             "(fraction of heads whose attention on the peak node exceeds eps=0.3). "
             "Intervals are 95% t across seeds.\n")

    partial = [r['run'] for r in runs if r.get('partial')]
    if partial:
        L.append(f"> **Partial runs included** (training snapshot ckpt.pt still present): {', '.join(partial)}\n")

    L.append("\n## Main table — sink rate and task score\n")
    L.append("| arm | n | sink rate | score | metric | sink consistency | top1/top2 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in sorted(arms, key=lambda a: -(stats([r.get('sink') for r in arms[a]]) or {'mean': -1})['mean']):
        rs = arms[arm]
        mn = next((r.get('metric_name') for r in rs if r.get('metric_name')), '?')
        L.append(f"| `{arm}` | {len(rs)} | {fmt(stats([r.get('sink') for r in rs]))} | "
                 f"{fmt(stats([r.get('score') for r in rs]))} | {mn} | "
                 f"{fmt(stats([r.get('sink_consistency') for r in rs]), 3)} | "
                 f"{fmt(stats([r.get('dominance') for r in rs]), 2)} |")

    L.append("\n## Sink ablation — is the sink load-bearing?\n")
    L.append("MAE after zeroing attention onto the named node. `sink/none` > 1 means the sink carries function.\n")
    L.append("| arm | n | none | sink | random | top-degree | sink2 | sink/none |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in sorted(arms):
        rs = [r for r in arms[arm] if r.get('abl_none') is not None]
        if not rs:
            continue
        L.append(f"| `{arm}` | {len(rs)} | " + " | ".join(
            fmt(stats([r.get(k) for r in rs]), 3) for k in
            ('abl_none', 'abl_sink', 'abl_random', 'abl_topdeg', 'abl_sink2')) +
            f" | {fmt(stats([r.get('abl_sink_ratio') for r in rs]), 2)} |")

    L.append("\n## Perturbation reach — final-layer response by graph distance\n")
    L.append("| arm | n | d0 | d1 | d2 | d3+ | d3+/d0 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in sorted(arms):
        rs = [r for r in arms[arm] if r.get('pert_d0') is not None]
        if not rs:
            continue
        L.append(f"| `{arm}` | {len(rs)} | " + " | ".join(
            fmt(stats([r.get(k) for r in rs]), 4) for k in
            ('pert_d0', 'pert_d1', 'pert_d2', 'pert_d3plus')) +
            f" | {fmt(stats([r.get('pert_reach') for r in rs]), 3)} |")

    cert_arms = {a: [r for r in rs if r.get('cert_observed') is not None] for a, rs in arms.items()}
    cert_arms = {a: rs for a, rs in cert_arms.items() if rs}
    if cert_arms:
        L.append("\n## Symmetry–locality certificate\n")
        L.append("A vacuous certificate (certified ~0 while observed is high) is the point: "
                 "structure cannot explain these sinks, so the scoring mechanism must.\n")
        L.append("| arm | n | certified no-sink | observed sink | median eps_1 | locality viol. | orbit viol. | orbit spread |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm in sorted(cert_arms):
            rs = cert_arms[arm]
            L.append(f"| `{arm}` | {len(rs)} | " + " | ".join(
                fmt(stats([r.get(k) for r in rs]), 3) for k in
                ('cert_certified', 'cert_observed', 'cert_eps1')) +
                f" | {max(r.get('cert_loc_viol') or 0 for r in rs)} "
                f"| {max(r.get('cert_orbit_viol') or 0 for r in rs)} "
                f"| {fmt(stats([r.get('cert_orbit_spread') for r in rs]), 6)} |")

    probe_arms = {a: [r for r in rs if r.get('probe_max_dev') is not None] for a, rs in arms.items()}
    probe_arms = {a: rs for a, rs in probe_arms.items() if rs}
    if probe_arms:
        L.append("\n## Equivariance probe on vertex-transitive graphs\n")
        L.append("Theory forces s_j == 1/n exactly for every node in an orbit. Vnode arms are "
                 "probed on the augmented cycle: there `deviation` is the within-orbit spread of "
                 "the real nodes (still forced to zero) and `s_vnode` is the attention received "
                 "by the vnode — the singleton orbit where the barrier permits concentration.\n")
        L.append("| arm | n | PE | max deviation (real-node orbit) | max s | s_vnode (worst) |")
        L.append("|---|---|---|---|---|---|")
        for arm in sorted(probe_arms, key=lambda a: -max(r['probe_max_dev'] for r in probe_arms[a])):
            rs = probe_arms[arm]
            pe = next((r.get('probe_pe') for r in rs if r.get('probe_pe')), '?')
            sv = stats([r.get('probe_s_vnode') for r in rs])
            L.append(f"| `{arm}` | {len(rs)} | {pe} | "
                     f"{max(r['probe_max_dev'] for r in rs):.3e} | "
                     f"{fmt(stats([r.get('probe_max_s') for r in rs]), 4)} | "
                     f"{fmt(sv, 3) if sv else '--'} |")

    md = "\n".join(L) + "\n"
    sys.stdout.write(md)
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(md)
        os.chmod(args.out, 0o664)
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'runs': runs,
                       'arms': {a: {'n': len(rs), 'runs': [r['run'] for r in rs]}
                                for a, rs in arms.items()}}, fh, indent=2)
        os.chmod(args.json, 0o664)


if __name__ == '__main__':
    main()
