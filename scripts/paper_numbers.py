"""Every number in the paper, computed from committed run outputs.

Writes, into --out-dir:
  numbers.tex   \\newcommand macros for every inline number (\\SRgrit, \\MAEgrit, ...)
  tab_*.tex     appendix table bodies (rows only; the paper owns the tabular frame)
and, with --md PATH, a human-readable dump of the same values to PATH.
Seeded runs only (<arm>-s<k>); intervals are 95% t-intervals over seeds, as in
aggregate_results.py. A missing arm yields a \\pending{...}
placeholder instead of a number.

Usage: python scripts/paper_numbers.py [--outputs outputs] --out-dir ../paper_numbers
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262}
EPS = 0.3
ARMS = {   # macro key -> outputs prefix (ZINC)
    'grit': 'zinc-grit', 'gritdot': 'zinc-grit-dot', 'dotbias': 'zinc-grit-dotbias',
    'gps': 'zinc-graphgps', 'gpsnone': 'zinc-gps-none', 'gpsgated': 'zinc-gps-gatedgcn',
    'lappe': 'zinc-gps-lappe-none', 'gpsvn': 'zinc-gps-vnode', 'gpsvnpe': 'zinc-gps-vnode-perecomp',
    'gritvn': 'zinc-grit-vnode', 'gritvnbias': 'zinc-grit-vnode-bias',
    'gritvnpe': 'zinc-grit-vnode-perecomp', 'entone': 'zinc-grit-forced-ent1',
    'enttwo': 'zinc-grit-forced-ent2', 'dotlogit': 'zinc-grit-dotlogit', 'blind': 'zinc-grit-blind',
    'noclamp': 'zinc-grit-noclamp', 'official': 'zinc-grit-official', 'plusdot': 'zinc-grit-plusdot',
    'gritL': 'zinc-grit-L20', 'dotlogitL': 'zinc-grit-dotlogit-L20',
}
LABEL = {
    'grit': 'GRIT', 'dotlogit': 'GRIT-dotlogit', 'blind': 'GRIT-blind', 'gritdot': 'GRIT-dot',
    'dotbias': 'GRIT-dotbias', 'plusdot': 'GRIT-plusdot', 'noclamp': 'GRIT-noclamp',
    'official': 'GRIT-official', 'entone': 'GRIT-entropy-1', 'enttwo': 'GRIT-entropy-2',
    'gritvn': 'GRIT + vnode', 'gritvnbias': 'GRIT + vnode + bias', 'gritvnpe': 'GRIT + vnode (PE rec.)',
    'gps': 'GraphGPS (GINE)', 'gpsgated': 'GPS-GatedGCN', 'gpsnone': 'GPS-none',
    'lappe': 'Transformer + LapPE', 'gpsvn': 'GPS + vnode', 'gpsvnpe': 'GPS + vnode (PE rec.)',
    'gritL': 'GRIT, 20 layers', 'dotlogitL': 'GRIT-dotlogit, 20 layers',
}
FULL_ORDER = ['grit', 'dotlogit', 'blind', 'gritdot', 'dotbias', 'plusdot', 'noclamp', 'official',
              'entone', 'enttwo', 'gritvn', 'gritvnbias', 'gritvnpe', 'gps', 'gpsgated', 'gpsnone',
              'lappe', 'gpsvn', 'gpsvnpe', 'gritL', 'dotlogitL']


def rnd(x, dec):
    q = Decimal(1).scaleb(-dec)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def ci_half(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float('nan')
    return T95.get(len(v) - 1, 1.96) * v.std(ddof=1) / math.sqrt(len(v))


def pm(v, dec=3):
    """'m \\pm h' over seeds; a single value prints alone; none prints a placeholder."""
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    if len(v) == 1:
        return rnd(v[0], dec)
    return f'{rnd(np.mean(v), dec)} \\pm {rnd(ci_half(v), dec)}'


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def peak(agg, key='overall_sink_rate'):
    m = agg[key]['mean']
    vals = [(v, i) for i, v in enumerate(m) if v == v]
    return max(vals)[1] if vals else None


class Out:
    def __init__(self):
        self.macros = {}
        self.md = []

    def m(self, name, value, note=''):
        assert re.fullmatch(r'[A-Za-z]+', name), name
        self.macros[name] = value if value is not None else '\\pending{--}'
        self.md.append(f'- `\\{name}` = {self.macros[name]}  {note}')


def seeded_runs(outputs, prefix):
    return sorted(d for d in glob.glob(os.path.join(outputs, prefix + '-s[0-9]'))
                  if os.path.isfile(os.path.join(d, 'eval_summary.json'))
                  and os.path.isfile(os.path.join(d, 'metrics.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--md', default=None, help='also write a readable dump here')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    O = Out()
    tabs = {}

    # ---------------------------------------------------------------- per-arm ZINC
    arm = {}
    for k, pre in ARMS.items():
        runs = seeded_runs(args.outputs, pre)
        r = {'runs': runs, 'sr': [], 'mae': [], 'ns': [], 'abl': defaultdict(list),
             'ablnr': defaultdict(list)}
        for d in runs:
            agg = load(os.path.join(d, 'eval_summary.json'))['agg']
            met = load(os.path.join(d, 'metrics.json'))
            p = peak(agg)
            r['sr'].append(agg['overall_sink_rate']['mean'][p])
            r['ns'].append(max(v for v in agg['norm_sink_score']['mean'] if v == v))  # threshold-free: own max
            r['mae'].append(met['test_mae'])
            for fn, key in (('sink_ablation_s0.json', 'abl'), ('sink_ablation_norenorm_s0.json', 'ablnr')):
                a = load(os.path.join(d, fn))
                if a:
                    c = a['conditions']
                    for cond in ('none', 'sink', 'random'):
                        r[key][cond].append(c[cond]['mae'])
                    r[key]['s/n'].append(c['sink']['mae'] / c['none']['mae'])
                    r[key]['r/n'].append(c['random']['mae'] / c['none']['mae'])
                    r[key]['s/r'].append(c['sink']['mae'] / c['random']['mae'])
        arm[k] = r
        O.m(f'SR{k}', pm(r['sr']), f'sink rate, n={len(runs)}')
        O.m(f'MAE{k}', pm(r['mae']), 'test MAE')
        O.m(f'N{k}', str(len(runs)) if runs else None, 'seeds')
        O.m(f'NS{k}', pm(r['ns'], 1), 'n * max s_j, max over layers')
        O.m(f'SRM{k}', rnd(np.mean(r['sr']), 2) if r['sr'] else None, 'sink rate, mean only')
        O.m(f'MAEM{k}', rnd(np.mean(r['mae']), 3) if r['mae'] else None, 'MAE, mean only')
        for key, tag in (('ablnr', 'AB'), ('abl', 'ABR')):
            O.m(f'{tag}sr{k}', pm(r[key]['s/r'], 2), f'{key} sink/random')
            O.m(f'{tag}sn{k}', pm(r[key]['s/n'], 2), f'{key} sink/none')
            O.m(f'{tag}rn{k}', pm(r[key]['r/n'], 2), f'{key} random/none')

    DOTARMS = ('gritdot', 'dotbias', 'gps', 'gpsnone', 'gpsgated', 'lappe', 'gpsvn', 'gpsvnpe', 'dotlogit', 'plusdot')
    dm = [np.mean(arm[k]['sr']) for k in DOTARMS if arm[k]['sr']]
    O.m('SRdotmin', rnd(min(dm), 2), 'min arm-mean sink rate over dot-product arms')
    O.m('SRdotmax', rnd(max(dm), 2), 'max arm-mean sink rate over dot-product arms')
    sa = [np.mean(arm[k]['ablnr']['s/r']) for k in DOTARMS if arm[k]['ablnr']['s/r']]
    O.m('ABsrdotmin', rnd(min(sa), 2), 'min arm-mean no-renorm sink/random over dot-product arms')
    O.m('ABsrdotmax', rnd(max(sa), 2), 'max arm-mean no-renorm sink/random over dot-product arms')
    # full ZINC table: arm & n & SR & MAE & n*s & no-renorm s/n r/n s/r & renorm s/r
    rows = []
    for k in FULL_ORDER:
        r = arm[k]
        if not r['runs']:
            continue
        cells = [LABEL[k], str(len(r['runs'])), pm(r['sr']), pm(r['mae']), pm(r['ns'], 1),
                 pm(r['ablnr']['s/n'], 2), pm(r['ablnr']['r/n'], 2), pm(r['ablnr']['s/r'], 2),
                 pm(r['abl']['s/r'], 2)]
        rows.append(' & '.join([cells[0], cells[1]] + [f'${c}$' if c else '--' for c in cells[2:]]) + r' \\')
    tabs['tab_full'] = rows

    # ---------------------------------------------------------------- epsilon sweep
    rows = []
    for k in ('grit', 'gritvn', 'entone', 'gritdot', 'dotbias', 'gps', 'gpsnone', 'gpsvn', 'lappe',
              'dotlogit', 'blind', 'noclamp', 'official'):
        r = arm[k]
        if not r['runs']:
            continue
        vals = []
        for e in ('0.1', '0.2', '0.3', '0.5', '0.8'):
            per = []
            for d in r['runs']:
                agg = load(os.path.join(d, 'eval_summary.json'))['agg']
                m = [v for v in agg[f'sink_rate_eps{e}']['mean'] if v == v]
                per.append(max(m))
            vals.append(rnd(np.mean(per), 2))
        rows.append(f'{LABEL[k]} & ' + ' & '.join(vals) + f' & ${pm(r["ns"], 1)}$ \\\\')
    tabs['tab_eps'] = rows
    # largest eps up to which GRIT's mean sink rate is below every dot-product arm's mean
    DOT = ('gritdot', 'dotbias', 'gps', 'gpsnone', 'gpsgated', 'lappe', 'gpsvn', 'gpsvnpe', 'dotlogit', 'plusdot')
    ok = None
    for e in ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'):
        def sr_e(k):
            return np.mean([max(v for v in load(os.path.join(d, 'eval_summary.json'))['agg'][f'sink_rate_eps{e}']['mean']
                                if v == v) for d in arm[k]['runs']])
        g = sr_e('grit')
        if all(g < sr_e(k) for k in DOT if arm[k]['runs']):
            ok = e
        else:
            break
    O.m('EPSmaxok', ok, "largest eps (from 0.1) up to which GRIT < every dot-product arm")

    # ---------------------------------------------------------------- Peptides
    def pep(prefix):
        runs = seeded_runs(args.outputs, prefix)
        sr, apv, ms = [], [], []
        for d in runs:
            agg = load(os.path.join(d, 'eval_summary.json'))['agg']
            met = load(os.path.join(d, 'metrics.json'))
            p = peak(agg)
            sr.append(agg['overall_sink_rate']['mean'][p])
            ms.append(max(v for v in agg['max_sink_score']['mean'] if v == v))
            apv.append(met.get('test_ap'))
        return runs, sr, apv, ms
    for i, q in enumerate('ABCD', start=1):
        runs, sr, apv, ms = pep(f'peptides-grit-Q{i}m')
        O.m(f'SRpepQm{q}', pm(sr), f'matched quartile Q{i}m, n={len(runs)}')
        O.m(f'APpepQm{q}', pm(apv), 'AP')
        O.m(f'MSpepQm{q}', pm(ms), 'max over layers of mean max_j s_j')
        runs, sr, apv, ms = pep(f'peptides-grit-Q{i}')
        O.m(f'SRpepQ{q}', pm(sr), f'unmatched quartile Q{i}, n={len(runs)}')
        O.m(f'APpepQ{q}', pm(apv), 'AP')
    runs, sr, apv, ms = pep('peptides-grit')
    O.m('SRpep', pm(sr), f'full-data GRIT, n={len(runs)}')
    O.m('APpep', pm(apv), 'AP')
    runs_dl, sr_dl, ap_dl, _ = pep('peptides-grit-dotlogit')
    O.m('SRpepdotlogit', pm(sr_dl), f'full-data GRIT-dotlogit, n={len(runs_dl)}')
    O.m('APpepdotlogit', pm(ap_dl), 'AP')

    bins = [(0, 91), (91, 136), (136, 204), (204, 300)]
    rows = []
    for prefix, lab in (('peptides-grit', 'GRIT'), ('peptides-grit-dotlogit', 'GRIT-dotlogit')):
        per = {b: {'sr': [], 'ms': [], 'ns': [], 'G': 0} for b in bins}
        for d in seeded_runs(args.outputs, prefix):
            with open(os.path.join(d, 'eval_suite.pkl'), 'rb') as f:
                res = pickle.load(f)['results']
            layers = [l for l in sorted(res) if l > 0]
            for b in bins:
                idx = [i for i, m in enumerate(res[layers[0]]) if b[0] < m['num_nodes'] <= b[1]]
                if not idx:
                    continue
                per[b]['G'] = len(idx)
                per[b]['sr'].append(max(np.mean([res[l][i]['sink_rate_eps0.3'] for i in idx]) for l in layers))
                per[b]['ms'].append(max(np.mean([res[l][i]['max_sink_score'] for i in idx]) for l in layers))
                per[b]['ns'].append(max(np.mean([res[l][i]['norm_sink_score'] for i in idx]) for l in layers))
        if lab == 'GRIT':
            for q, b in zip('ABCD', bins):
                O.m(f'SRbin{q}', pm(per[b]['sr']), f'size bin {b}, G={per[b]["G"]}')
                O.m(f'MSbin{q}', pm(per[b]['ms']), 'max s')
                O.m(f'NSbin{q}', pm(per[b]['ns'], 1), 'n * max s')
        if any(per[b]['sr'] for b in bins):
            for key, nm, dec in (('sr', 'sink rate', 3), ('ms', 'max $s_j$', 3), ('ns', '$n\\max_j s_j$', 1)):
                rows.append(f'{lab} & {nm} & ' + ' & '.join(f'${pm(per[b][key], dec)}$' for b in bins) + r' \\')
    tabs['tab_sizebins'] = rows

    # ---------------------------------------------------------------- over-mixing
    rows = []
    ray_all = {}
    for k in ('grit', 'dotlogit', 'blind', 'gritdot', 'dotbias', 'gps', 'gpsgated', 'gpsnone', 'lappe', 'gpsvn',
              'gpsvnpe', 'gritvn', 'gritvnbias', 'gritvnpe', 'entone', 'enttwo', 'gritL', 'dotlogitL'):
        r = arm[k]
        if not r['runs']:
            continue
        ray, an0, an1, cr0, cr1 = [], [], [], [], []
        for d in r['runs']:
            agg = load(os.path.join(d, 'eval_summary.json'))['agg']
            m = agg['dirichlet_rayleigh']['mean']
            ray.append(m[-1] / m[0])
            an0.append(agg['anisotropy']['mean'][0]); an1.append(agg['anisotropy']['mean'][-1])
            cr = load(os.path.join(d, 'eval_summary_cr.json')) or {'agg': agg}
            if 'consensus_residual' in cr['agg']:
                cr0.append(cr['agg']['consensus_residual']['mean'][0])
                cr1.append(cr['agg']['consensus_residual']['mean'][-1])
        rows.append(f'{LABEL[k]} & ${pm(ray, 2)}$ & ${pm(an0, 2)}$ & ${pm(an1, 2)}$ & '
                    f'${pm(cr0, 2) or "--"}$ & ${pm(cr1, 2) or "--"}$ \\\\')
        O.m(f'RAY{k}', pm(ray, 2), 'Dirichlet Rayleigh last/first layer')
        ray_all[k] = ray
        O.m(f'CRfirst{k}', pm(cr0, 2), 'consensus residual, layer 0')
        O.m(f'CRlast{k}', pm(cr1, 2), 'consensus residual, last layer')
    tabs['tab_mixing'] = rows
    below = sorted(k for k, v in ray_all.items() if min(v) < 1)
    O.m('RAYbelow', ', '.join(LABEL[k] for k in below) if below else 'none',
        'arms with any seed whose Dirichlet Rayleigh ratio last/first < 1')
    rest = [np.mean(v) for k, v in ray_all.items() if k != 'gritvnpe']
    O.m('RAYminrest', rnd(min(rest), 2), 'min arm-mean ratio, all arms but GRIT+VN (PE rec.)')
    O.m('RAYmaxrest', rnd(max(rest), 2), 'max arm-mean ratio')
    if 'gritvnpe' in ray_all:
        O.m('RAYvnpe', f"{rnd(min(ray_all['gritvnpe']), 2)}--{rnd(max(ray_all['gritvnpe']), 2)}", 'GRIT+VN (PE rec.) per-seed range')

    # norm growth (squared max/mean node norm), GRIT
    g0, g1 = [], []
    for d in arm['grit']['runs']:
        agg = load(os.path.join(d, 'eval_summary.json'))['agg']
        g0.append(agg['max_to_mean_ratio']['mean'][0]); g1.append(agg['max_to_mean_ratio']['mean'][-1])
    O.m('NORMfirst', pm(g0, 1), 'GRIT max_j ||h_j||^2 / mean ||h||^2, input layer')
    O.m('NORMlast', pm(g1, 1), 'GRIT, last layer')

    # ---------------------------------------------------------------- saturation (GRIT, _cr)
    up, rz, lin, up1, rz1 = [], [], [], [], []
    rows = []
    for k in ('grit', 'gritvn', 'entone', 'dotlogit', 'blind', 'official', 'noclamp'):
        per_layer = defaultdict(lambda: ([], [], []))
        for d in arm[k]['runs']:
            cr = load(os.path.join(d, 'eval_summary_cr.json')) or load(os.path.join(d, 'eval_summary.json'))
            a = cr['agg']
            if 'clamp_upper_frac' not in a:
                continue
            u, z = a['clamp_upper_frac']['mean'], a['relu_zero_frac']['mean']
            for l in range(1, len(u)):
                per_layer[l][0].append(u[l]); per_layer[l][1].append(z[l]); per_layer[l][2].append(1 - u[l] - z[l])
        if not per_layer:
            continue
        L2 = [l for l in per_layer if l >= 2]
        u_all = [x for l in L2 for x in per_layer[l][0]]
        z_all = [x for l in L2 for x in per_layer[l][1]]
        n_all = [x for l in L2 for x in per_layer[l][2]]
        rows.append(f'{LABEL[k]} & {rnd(np.mean(per_layer[1][0]), 2)} / {rnd(np.mean(per_layer[1][1]), 2)} & '
                    f'{rnd(min(u_all), 2)}--{rnd(max(u_all), 2)} & {rnd(min(z_all), 2)}--{rnd(max(z_all), 2)} & '
                    f'{rnd(min(n_all), 3)}--{rnd(max(n_all), 3)} \\\\')
        if k == 'grit':
            O.m('SATup', f'{rnd(min(u_all), 2)}--{rnd(max(u_all), 2)}', 'GRIT layers 2-10, every seed: score >= 5')
            O.m('SATzero', f'{rnd(min(z_all), 2)}--{rnd(max(z_all), 2)}', 'score <= 0 (ReLU zero)')
            O.m('SATlin', f'{rnd(100 * max(n_all), 0)}', 'max % of coordinates in (0,5), layers 2-10')
            O.m('SATends', f'{rnd(100 * (1 - max(n_all)), 0)}', 'min % at 0 or 5, layers 2-10')
    tabs['tab_saturation'] = rows

    # ---------------------------------------------------------------- no-op
    rows = []
    for k in ('grit', 'gritdot', 'dotbias', 'gps', 'gpsnone', 'lappe', 'gpsvn', 'gpsvnpe', 'gritvn',
              'dotlogit', 'blind'):
        rp, pvn, rvn, rreal = [], [], [], []
        for d in arm[k]['runs']:
            agg = load(os.path.join(d, 'eval_summary.json'))['agg']
            p = peak(agg)
            if not os.path.isfile(os.path.join(d, 'eval_suite.pkl')):   # not in the public release
                continue
            with open(os.path.join(d, 'eval_suite.pkl'), 'rb') as f:
                g = [m for m in pickle.load(f)['results'][p] if 'sink_value_ratio' in m]
            yes = [m['sink_value_ratio'] for m in g if m['overall_sink_rate'] > 0]
            if yes:
                rp.append(float(np.median(yes)))
            if 'vnode' in d:
                vn = [m['sink_value_ratio'] for m in g if m.get('vnode_is_sink') == 1.0]
                re_ = [m['sink_value_ratio'] for m in g if m.get('vnode_is_sink') == 0.0]
                pvn.append(len(vn) / len(g))
                if vn:
                    rvn.append(float(np.median(vn)))
                if re_:
                    rreal.append(float(np.median(re_)))
        if not rp:
            continue
        O.m(f'VR{k}', pm(rp, 2), 'median value ratio of the argmax node, sink-present graphs, peak layer')
        if pvn:
            O.m(f'PVN{k}', pm(pvn, 2), 'P(VN is the argmax) at the peak layer')
            O.m(f'VRvn{k}', pm(rvn, 2), 'value ratio when the VN is the argmax')
            O.m(f'VRreal{k}', pm(rreal, 2), 'value ratio when a real atom is the argmax')
        rows.append(f'{LABEL[k]} & ${pm(rp, 2)}$ & ' + (f'${pm(pvn, 2)}$ & ' + (f'${pm(rvn, 2)}$' if pm(rvn, 2) else '--') + f' & ${pm(rreal, 2)}$'
                                                      if pvn else '-- & -- & --') + r' \\')
    tabs['tab_noop'] = rows

    # ---------------------------------------------------------------- norm intervention
    rows = []
    rhos = ('1.0', '2.0', '8.0', '32.0')
    for k in ('grit', 'noclamp', 'official', 'blind', 'gritvn', 'dotlogit', 'gritdot', 'dotbias', 'gps', 'gpsnone',
              'gpsvn'):
        per = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for d in arm[k]['runs']:
            x = load(os.path.join(d, 'norm_intervention.json'))
            if not x:
                continue
            for lay, byrho in x['layers'].items():
                for rho in rhos:
                    per[lay][rho][0].append(byrho[rho]['ns_median'])
                    per[lay][rho][1].append(byrho[rho]['frac_heads_above_eps_mean'])
        if not per:
            continue
        for lay in sorted(per, key=int):
            rows.append(f'{LABEL[k]} & {lay} & ' + ' & '.join(
                f'{rnd(np.mean(per[lay][r][0]), 2)} / {rnd(100 * np.mean(per[lay][r][1]), 1)}' for r in rhos) + r' \\')
        hf32 = [np.mean(per[l]['32.0'][1]) for l in per]
        ns32 = [np.mean(per[l]['32.0'][0]) for l in per]
        hf2 = [np.mean(per[l]['2.0'][1]) for l in per]
        hf8 = [np.mean(per[l]['8.0'][1]) for l in per]
        O.m(f'NIhfmax{k}', rnd(100 * max(hf32), 1), '% heads > eps at rho=32, max over layers 3/5/8')
        O.m(f'NInsmax{k}', rnd(max(ns32), 1), 'median n*s at rho=32, max over layers')
        O.m(f'NIhftwo{k}', f'{rnd(100 * min(hf2), 0)}--{rnd(100 * max(hf2), 0)}', '% heads > eps at rho=2, layer range')
        O.m(f'NIhfeight{k}', f'{rnd(100 * min(hf8), 0)}--{rnd(100 * max(hf8), 0)}', '% heads > eps at rho=8')
        O.m(f'NIhftwomax{k}', rnd(100 * max(hf2), 1), '% heads > eps at rho=2, max over layers')
        O.m(f'NIhfeightmax{k}', rnd(100 * max(hf8), 1), '% heads > eps at rho=8, max over layers')
    tabs['tab_normint'] = rows
    hf8_dot = []
    for k in ('dotlogit', 'gritdot', 'dotbias', 'gps', 'gpsnone', 'gpsvn'):
        vals = []
        for d in arm[k]['runs']:
            x = load(os.path.join(d, 'norm_intervention.json'))
            if x:
                vals.append(min(v['8.0']['frac_heads_above_eps_mean'] for v in x['layers'].values()))
        if vals:
            hf8_dot.append(np.mean(vals))
    O.m('NIhfeightmindot', rnd(100 * min(hf8_dot), 0) if hf8_dot else None,
        '% heads > eps at rho=8: min over dot-product arms (and over layers 3/5/8; seed mean of per-seed minima)')

    # ---------------------------------------------------------------- LM reference
    lm = load(os.path.join(args.outputs, 'lm_reference', 'lm_sink_reference.json'))
    rows = []
    if lm:
        names = {'EleutherAI/pythia-70m': 'Pythia-70M', 'EleutherAI/pythia-160m': 'Pythia-160M', 'gpt2': 'GPT-2 small'}
        srs = defaultdict(list)
        for mname, byT in lm['models'].items():
            c = []
            for T in ('23', '269'):
                x = byT[T]
                srs[T].append(x['sink_rate'])
                c += [rnd(x['sink_rate'], 2), rnd(x['n_times_max_s_peak'], 1)]
            rows.append(f'{names.get(mname, mname)} & ' + ' & '.join(c) + r' \\')
        O.m('LMsrA', f'{rnd(min(srs["23"]), 2)}--{rnd(max(srs["23"]), 2)}', 'LM sink rate at T=23')
        O.m('LMsrB', f'{rnd(min(srs["269"]), 2)}--{rnd(max(srs["269"]), 2)}', 'LM sink rate at T=269')
    tabs['tab_lm'] = rows

    # ---------------------------------------------------------------- certificate and probe
    cert = defaultdict(list); eps1 = defaultdict(list); spread = defaultdict(list); viol = defaultdict(list)
    EQ = ('grit', 'gritdot', 'dotbias', 'gps', 'gpsnone', 'gpsgated', 'gpsvn', 'gpsvnpe', 'gritvn',
          'gritvnbias', 'gritvnpe', 'entone', 'enttwo', 'dotlogit', 'blind', 'noclamp', 'official')
    nhead = 0
    for k in EQ:
        for d in arm[k]['runs']:
            c = load(os.path.join(d, 'theory_certificate.json'))
            if not c:
                continue
            nhead += c['n_head_instances']
            cert[k].append(c['certified_no_sink_frac']); eps1[k].append(c['eps_r_median'][1])
            spread[k].append(c['max_within_orbit_spread']); viol[k].append(c['orbit_bound_violations'])
    allv = lambda dct, skip=(): [x for kk, v in dct.items() if kk not in skip for x in v]
    O.m('CERTmax', rnd(100 * max(allv(cert)), 1), 'max % of head instances certified no-sink, any run')
    O.m('CERTmaxrest', rnd(100 * max(allv(cert, skip=('grit',))), 1), '... excluding the GRIT arm')
    O.m('EPSonemin', rnd(min(allv(eps1, skip=('gritvnpe',))), 2), 'min median radius-1 leakage, all arms but GRIT+VN (PE rec.)')
    O.m('EPSonevnpe', f"{rnd(min(eps1['gritvnpe']), 2)}--{rnd(max(eps1['gritvnpe']), 2)}", 'GRIT+VN (PE rec.)')
    sp = allv(spread, skip=('dotbias',))
    O.m('SPREADmax', f'{max(sp):.1e}'.replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}',
        'worst within-orbit spread, equivariant-PE runs except GRIT-dotbias')
    O.m('SPREADdotbias', f"{rnd(max(spread['dotbias']), 3)}", 'GRIT-dotbias worst spread')
    O.m('ORBVIOLdotbias', str(int(sum(viol['dotbias']))), 'orbit-bound violations, GRIT-dotbias (all seeds)')
    O.m('ORBVIOLrest', str(int(sum(allv(viol, skip=('dotbias',))))), 'violations, other equivariant runs')
    O.m('NHEAD', f'{nhead:,}'.replace(',', '{,}'), 'head instances certified in total')

    def probe(k, fn='symmetry_probe.json'):
        dev, sv, mx = [], [], []
        for d in arm[k]['runs']:
            p = load(os.path.join(d, fn))
            if not p:
                continue
            dev.append(max(x['max_abs_deviation'] for x in p['sizes'].values()))
            mx.append(max(x['max_s'] for x in p['sizes'].values()))
            if p.get('vnode'):
                sv.append(max(x['s_vnode'] for x in p['sizes'].values()))
        return dev, sv, mx
    sci = lambda x: f'{x:.1e}'.replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}'
    dev_novn, dev_vn = [], []
    for k in ('grit', 'gritdot', 'gps', 'gpsnone', 'gpsgated', 'gpsvn', 'gpsvnpe', 'gritvn',
              'gritvnbias', 'gritvnpe', 'entone', 'enttwo', 'blind', 'noclamp', 'official'):
        dev, sv, mx = probe(k)
        (dev_vn if 'vn' in k else dev_novn).extend(dev)
        if sv:
            O.m(f'SV{k}', pm(sv, 3), 'probe: worst s_vnode per seed')
    O.m('PROBEdevmax', sci(max(dev_novn)), 'probe: worst deviation, equivariant-PE arms without VN, except dotbias')
    O.m('PROBEdevvn', sci(max(dev_vn)), 'probe: worst real-node deviation, VN arms')
    O.m('PROBEdevdotbias', sci(max(probe('dotbias')[0])), 'probe: GRIT-dotbias worst deviation')
    O.m('PROBEdevdotlogit', rnd(max(probe('dotlogit')[0]), 2), 'probe: GRIT-dotlogit worst deviation (float32)')
    fp = probe('dotbias', 'symmetry_probe_fp64.json')[0] + probe('dotlogit', 'symmetry_probe_fp64.json')[0]
    O.m('PROBEfpdot', sci(max(fp)) if fp else None, 'probe in float64: worst deviation, GRIT-dotbias and GRIT-dotlogit')
    fpb = probe('dotbias', 'symmetry_probe_fp64.json')[0]
    O.m('PROBEfpdotbias', sci(max(fpb)) if fpb else None, 'probe in float64: GRIT-dotbias worst deviation')
    dev, _, mx = probe('lappe')
    O.m('PROBElappe', pm(mx, 3), 'probe: LapPE max s')

    # ---------------------------------------------------------------- logit-range bound 5||a||_1
    try:
        import torch
        bnd = {}
        for k in ('grit', 'gritvn'):
            vals = []
            for d in arm[k]['runs']:
                sd = torch.load(os.path.join(d, 'best_model.pt'), map_location='cpu', weights_only=True)
                ws = sorted((kk for kk in sd if kk.endswith('mha.W_A.weight')),
                            key=lambda s: int(re.search(r'layers\.(\d+)\.', s).group(1)))
                vals += [5 * sd[w].abs().sum().item() for w in ws]
            bnd[k] = vals
        allv = bnd['grit']
        O.m('RANGEmin', rnd(min(allv), 2), "GRIT 5||a||_1, min over layers and seeds")
        O.m('RANGEmax', rnd(max(allv), 2), "GRIT 5||a||_1, max")
        off = [torch.load(os.path.join(d, 'best_model.pt'), map_location='cpu',
                          weights_only=True)['sink_bias_param'].item() for d in arm['gritvnbias']['runs']]
        O.m('VNbiasmin', rnd(min(off), 1), 'GRIT + vnode + bias: learned logit offset, min over seeds')
        O.m('VNbiasmax', rnd(max(off), 1), 'GRIT + vnode + bias: learned logit offset, max over seeds')
    except Exception as e:   # torch missing: leave placeholders
        print('range bound skipped:', e)
    O.m('DELTAzinc', rnd(math.log(22 * EPS / (1 - EPS)), 2), 'logit range a sink needs at n=23')
    O.m('DELTApep', rnd(math.log(268 * EPS / (1 - EPS)), 2), 'at n=269')

    # ---------------------------------------------------------------- write
    with open(os.path.join(args.out_dir, 'numbers.tex'), 'w') as f:
        f.write('% Generated by scripts/paper_numbers.py -- do not edit by hand.\n')
        for k, v in O.macros.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    for name, rows in tabs.items():
        with open(os.path.join(args.out_dir, f'{name}.tex'), 'w') as f:
            f.write('% Generated by scripts/paper_numbers.py -- do not edit by hand.\n')
            f.write('\n'.join(rows) + '\n')
    with open(args.md or os.devnull, 'w') as f:
        f.write('# Paper numbers\n\n' + '\n'.join(O.md) + '\n')
        for name, rows in tabs.items():
            f.write(f'\n## {name}\n\n' + '\n'.join(rows) + '\n')
    print(f'{len(O.macros)} macros, {len(tabs)} tables -> {args.out_dir}')


if __name__ == '__main__':
    main()
