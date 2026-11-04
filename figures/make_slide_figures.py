"""Figures Data de slides/comparaison.tex (anglais), style de slides/make_comparison_figures.py.

python figures/make_slide_figures.py  ->  figures/slides/data_*.png, à copier dans slides/figures/ sur main.
Noise 0.3 réunit noise_03 (200 époques) et noise_03_e100 (100 époques) : même run jusqu'à 60 000 mises à jour.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_data_figures import load_runs

OUT = 'figures/slides'
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 15, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
VARIANTS = (('Reference', ['data_ref'], MUTED), ('Noise 0.1', ['noise_01'], AQUA), ('Noise 0.3', ['noise_03', 'noise_03_e100'], AQUA),
            ('Selection 95%', ['sel_95'], BLUE), ('Selection 90%', ['sel_90'], BLUE), ('Selection 75%', ['sel_75'], BLUE),
            ('Selection 50%', ['sel_50'], BLUE))


def bars(ax, labels, values, colors, title, texts, xmax):
    ax.barh(labels, values, color=colors, height=0.68)
    for y, (v, t) in enumerate(zip(values, texts)):
        ax.annotate(t, (v, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
    ax.set(xlim=(0, xmax))
    ax.set_title(title, loc='left', fontsize=14, color=INK)
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)


def main():
    os.makedirs(OUT, exist_ok=True)
    img = Image.open('figures/data/kept_vs_rejected.png')
    img.resize((img.width * 4, img.height * 4), Image.NEAREST).save(f'{OUT}/data_kept_removed.png')

    runs = load_runs()
    labels, colors = [v[0] for v in VARIANTS], [v[2] for v in VARIANTS]
    groups = [sum((runs[n] for n in names), []) for _, names, _ in VARIANTS]
    ok = [[r for r in g if not r['collapsed']] for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for ax, (key, title, fmt) in zip(axes, (('fid', 'CNN FID', '{:.0f}'), ('precision', 'CNN precision', '{:.2f}'),
                                             ('recall', 'CNN recall', '{:.2f}'))):
        v = [np.mean([r['curves'][key][r['at']] for r in g]) for g in ok]
        bars(ax, labels, v, colors, title, [fmt.format(x) for x in v], max(v) * 1.35)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/data_results.png', dpi=200)

    fig, ax = plt.subplots(figsize=(7, 3.9))
    share = [100 * (len(g) - len(o)) / len(g) for g, o in zip(groups, ok)]
    bars(ax, labels, share, [ORANGE if l == 'Noise 0.3' else c for l, c in zip(labels, colors)], 'Collapsed runs (%)',
         [f'{len(g) - len(o)} / {len(g)}' for g, o in zip(groups, ok)], 75)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/data_collapse.png', dpi=200)
    for l, g, o in zip(labels, groups, ok):
        print(l, f'{len(g) - len(o)}/{len(g)}', 'FID', round(np.mean([r['curves']['fid'][r['at']] for r in o]), 1))


if __name__ == '__main__':
    main()
