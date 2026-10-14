"""Plot Figure 1 from scripts/make_fig1_data.py output.

Each atom is coloured by log2(n * s_j), its received attention relative to the uniform
share 1/n, at each model's peak-sinking layer (head average). White = uniform, red =
above, blue = below. The colourbar is drawn from the same colormap and norm as the atom
fills. One shared layout and one shared colour scale across panels.

Usage:
  python scripts/plot_fig1.py outputs/figures/fig1_data.json -o outputs/figures/fig1.pdf \
      [--panels GRIT GRIT-dotlogit GraphGPS] [--subtitles "gated score" ...] [--vlim 3]
"""
import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('-o', '--out', default='outputs/figures/fig1.pdf')
    ap.add_argument('--panels', nargs='+', default=None, help='panel labels to keep, in order (default: all)')
    ap.add_argument('--subtitles', nargs='+', default=None)
    ap.add_argument('--vlim', type=float, default=3.0, help='colour range ±log2(n s) (3 = x1/8..x8)')
    ap.add_argument('--width', type=float, default=5.5, help='inches (NeurIPS text width 5.5)')
    ap.add_argument('--height', type=float, default=2.0)
    args = ap.parse_args()

    with open(args.data) as f:
        d = json.load(f)
    g, panels, eps = d['graph'], d['panels'], d['eps']
    if args.panels:
        by_label = {p['label']: p for p in panels}
        panels = [by_label[k] for k in args.panels]
    n = g['num_nodes']
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ei = np.array(g['edge_index'])
    G.add_edges_from((int(a), int(b)) for a, b in ei.T if a != b)
    pos = nx.kamada_kawai_layout(G)

    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=-args.vlim, vcenter=0.0, vmax=args.vlim)
    k = len(panels)
    subs = args.subtitles or [''] * k
    assert len(subs) == k

    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(args.width, args.height))
    cb_w = 0.035
    gs = fig.add_gridspec(1, k + 1, width_ratios=[1] * k + [cb_w * k], wspace=0.08,
                          left=0.005, right=0.93, top=0.97, bottom=0.36)
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        s = np.array(p['s_peak_headmean'])
        c = np.log2(np.clip(n * s, 1e-12, None))
        j = p['argmax_node']
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.7, edge_color='0.55')
        others = [u for u in range(n) if u != j]
        ax.scatter([pos[u][0] for u in others], [pos[u][1] for u in others],
                   c=c[others], cmap=cmap, norm=norm, s=34, edgecolors='0.35',
                   linewidths=0.4, zorder=3)
        ax.scatter([pos[j][0]], [pos[j][1]], c=[c[j]], cmap=cmap, norm=norm, s=46,
                   edgecolors='black', linewidths=1.1, zorder=4)
        ax.set_aspect('equal')
        ax.axis('off')
        sr = p.get('graph_sink_rate', p.get('argmax_frac_heads_above_eps'))
        ax.annotate(p['label'], (0.5, 0.0), xycoords='axes fraction', xytext=(0, -2),
                    textcoords='offset points', ha='center', va='top',
                fontsize=7.5, fontweight='bold')
        ax.annotate(subs[i], (0.5, 0.0), xycoords='axes fraction', xytext=(0, -11),
                    textcoords='offset points', ha='center', va='top', fontsize=6.5,
                color='0.25')
        ax.annotate(
                f"layer {p['peak_layer']}: peak $n s_j$ = {n * p['argmax_s_headmean']:.1f}\n"
                f"{100 * sr:.0f}% of heads with $s_j>\\varepsilon$",
                (0.5, 0.0), xycoords='axes fraction', xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=5.8, color='0.25',
                linespacing=1.3)

    cax = fig.add_subplot(gs[0, k])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, extend='both')
    ticks = [t for t in range(-int(args.vlim), int(args.vlim) + 1)]
    cb.set_ticks(ticks)
    lab = {0: 'uniform\n$1/n$'}
    cb.set_ticklabels([lab.get(t, (f'$\\times{2 ** t}$' if t > 0 else f'$\\times 1/{2 ** -t}$'))
                       for t in ticks])
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    ye = math.log2(n * eps)
    if abs(ye) <= args.vlim:
        cb.ax.axhline(ye, color='black', lw=0.8)
    cb.set_label('$n\\,s_j$ (log scale)', fontsize=6, labelpad=1)
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.01)
    png = args.out.rsplit('.', 1)[0] + '.png'
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f'wrote {args.out} and {png}')


if __name__ == '__main__':
    main()
