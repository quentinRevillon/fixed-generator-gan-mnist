"""Figures Data de slides/comparaison.tex (anglais), style de slides/make_comparison_figures.py.

python figures/make_slide_figures.py  ->  figures/slides/data_*.png, à copier dans slides/figures/ sur main.
Noise 0.3 réunit noise_03 (200 époques) et noise_03_e100 (100 époques) : même run jusqu'à 60 000 mises à jour.
EXPECTED : seeds encore sur Juliet, comptés non effondrés dans data_collapse.png. À vider une fois les runs rapatriés.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from make_data_figures import load_runs

OUT = 'figures/slides'
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 15, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
VARIANTS = (('Reference', ['data_ref'], MUTED), ('Noise 0.1', ['noise_01'], AQUA), ('Noise 0.3', ['noise_03', 'noise_03_e100'], AQUA),
            ('Selection 95%', ['sel_95'], BLUE), ('Selection 90%', ['sel_90'], BLUE),
            ('Selection 50%', ['sel_50'], BLUE))
EXPECTED = {'Noise 0.3': 8}


def bars(ax, labels, values, colors, title, texts, xmax):
    ax.barh(labels, values, color=colors, height=0.68)
    for y, (v, t) in enumerate(zip(values, texts)):
        ax.annotate(t, (v, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
    ax.set(xlim=(0, xmax))
    ax.set_title(title, loc='left', fontsize=14, color=INK)
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)


def side_by_side(paths, out, gap=24):
    ims = [Image.open(p).convert('RGB') for p in paths]
    w, h = ims[0].size
    c = Image.new('RGB', (len(ims) * w + (len(ims) - 1) * gap, h), SURFACE)
    for i, im in enumerate(ims):
        c.paste(im, (i * (w + gap), 0))
    c.resize((c.width * 2, c.height * 2), Image.NEAREST).save(out)


def noise_steps():
    """Ce que voit D avec sigma_0 = 0.3 (images dans [-1, 1]) : mêmes images aux époques 1, 34, 67 et 100."""
    from evaluate import mnist
    x, y = mnist(train=True)
    idx = torch.stack([(y == c).nonzero()[0, 0] for c in (0, 3, 7)])
    eps = torch.randn((3, 1, 28, 28), generator=torch.Generator().manual_seed(0))
    fig, axes = plt.subplots(3, 4, figsize=(6, 4.9))
    for j, (epoch, sigma) in enumerate(((1, .3), (34, .2), (67, .1), (100, 0.))):
        for i in range(3):
            img = (x[idx[i]].view(1, 28, 28) * 2 - 1 + sigma * eps[i]).clamp(-1, 1)
            axes[i, j].imshow(img[0], cmap='gray', vmin=-1, vmax=1)
            axes[i, j].axis('off')
        axes[0, j].set_title(f'epoch {epoch}\n$\\sigma = {sigma:g}$', fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(f'{OUT}/data_noise_steps.png', dpi=200)


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
    n = [max(len(g), EXPECTED.get(l, 0)) for l, g in zip(labels, groups)]
    k = [len(g) - len(o) for g, o in zip(groups, ok)]
    bars(ax, labels, [100 * a / b for a, b in zip(k, n)], [ORANGE if l == 'Noise 0.3' else c for l, c in zip(labels, colors)],
         'Collapsed runs (%)', [f'{a} / {b}' for a, b in zip(k, n)], 75)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/data_collapse.png', dpi=200)
    side_by_side(['runs/data_ref_s0/grid.png', 'runs/data_ref_s4/grid.png'], f'{OUT}/data_collapse_samples.png')
    noise_steps()
    for l, g, o in zip(labels, groups, ok):
        print(l, f'{len(g) - len(o)}/{len(g)}', 'FID', round(np.mean([r['curves']['fid'][r['at']] for r in o]), 1))


if __name__ == '__main__':
    main()
