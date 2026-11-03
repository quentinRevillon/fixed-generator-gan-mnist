"""Figures de la baseline pour les slides, à taille de slide (dans slides/figures/).

Lit runs/baseline_200ep/metrics.jsonl et trois générateurs (époques 50, 120 et 200).

Utilisation, depuis la racine du repo :
    python slides/make_figures.py --weights <dossier contenant G_ep50.pth, G_ep120.pth, G_ep200.pth>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.ticker import ScalarFormatter

from evaluate import to_unit_images
from model import Generator
from utils import load_model

OUT = 'slides/figures'
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
EPOCHS = (50, 120, 200)
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def load_evaluations():
    """Évaluations (FID, P, R, classes) de la baseline à 200 époques."""
    with open('runs/baseline_200ep/metrics.jsonl') as f:
        return [r for r in map(json.loads, f) if 'fid' in r]


def mark(ax, epoch, label):
    """Trait vertical pointillé annoté."""
    ax.axvline(epoch, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate(label, (epoch, 1), xycoords=('data', 'axes fraction'), xytext=(4, -2), textcoords='offset points',
                va='top', fontsize=10, color=MUTED)


def plot_curves(rows):
    """FID (log) d'un côté, précision et rappel de l'autre : deux échelles, donc deux panneaux."""
    epochs = [r['epoch'] for r in rows]
    best = min(rows, key=lambda r: r['fid'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    a.plot(epochs, [r['fid'] for r in rows], color=BLUE, linewidth=2, marker='o', markersize=4)
    a.set(yscale='log', xlabel='Époque', ylabel='FID (échelle log)', xlim=(0, 205))
    a.yaxis.set_major_formatter(ScalarFormatter())
    a.grid(axis='y', color=GRID)
    a.annotate(f"minimum : {best['fid']:.0f}\n(époque {best['epoch']})", (best['epoch'], best['fid']), xytext=(0, 26),
               textcoords='offset points', ha='center', color=INK, fontsize=10)
    fifty = next(r for r in rows if r['epoch'] == 50)
    a.annotate(f"{fifty['fid']:.0f}", (50, fifty['fid']), xytext=(7, 3), textcoords='offset points', color=INK, fontsize=10)
    mark(a, 50, 'époque 50')
    for key, label, color, shape, offset in (('precision', 'Précision', BLUE, 'o', 7), ('recall', 'Rappel', ORANGE, 's', -17)):
        values = [r[key] for r in rows]
        b.plot(epochs, values, color=color, linewidth=2, marker=shape, markersize=4, label=label)
        b.annotate(label, (epochs[-1], values[-1]), xytext=(0, offset), textcoords='offset points', ha='right', color=INK, fontsize=10)
    b.set(xlabel='Époque', ylabel='Valeur', xlim=(0, 205), ylim=(0, 1))
    b.grid(axis='y', color=GRID)
    b.legend(loc='lower right', frameon=False, fontsize=10, labelcolor=MUTED)
    mark(b, 50, 'époque 50')
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_curves.pdf')


@torch.no_grad()
def plot_grids(rows, weights):
    """Mêmes z pour les trois générateurs : seule l'époque change."""
    z = torch.randn(36, 100, generator=torch.Generator().manual_seed(0))
    fid = {r['epoch']: r['fid'] for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for ax, epoch in zip(axes, EPOCHS):
        G = load_model(Generator(784), os.path.join(weights, f'G_ep{epoch}.pth'), 'cpu').eval()
        grid = torchvision.utils.make_grid(to_unit_images(G(z)), nrow=6, padding=2, pad_value=1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f'Époque {epoch} (FID {fid[epoch]:.0f})', fontsize=11, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_grids.png', dpi=200)


def plot_classes(rows):
    """Part de chaque chiffre aux époques 50, 120 et 200 : légende et formes pour ne pas dépendre de la couleur."""
    by_epoch = {r['epoch']: 100 * np.array(r['classes']) / sum(r['classes']) for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    width = 0.27
    for i, (epoch, color, hatch) in enumerate(zip(EPOCHS, (AQUA, BLUE, ORANGE), ('', '', '//'))):
        ax.bar(np.arange(10) + (i - 1) * width, by_epoch[epoch], width * 0.92, color=color, hatch=hatch,
               edgecolor=SURFACE, linewidth=0.5, label=f'Époque {epoch}')
    ax.axhline(10, color=MUTED, linestyle=':', linewidth=1.2)
    ax.annotate('réel : ≈ 10 %', (9.55, 10), xytext=(0, 3), textcoords='offset points', ha='left', fontsize=9, color=MUTED)
    ax.set(xticks=range(10), xlabel='Chiffre prédit par le classifieur', ylabel='% des images', ylim=(0, 27), xlim=(-0.7, 11))
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper right', labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_classes.png', dpi=200)


def plot_noe_ablation():
    """Recompose l'ablation de Noé à partir de ses grilles, avec des titres lisibles sur une slide (FID de son results.csv)."""
    runs = (('baseline_rerun', 'Baseline', '188'), ('train_rpgan', 'RpGAN seul (10 ép.)', '2619'),
            ('train_rpgan_r1', 'RpGAN + R1', '44'), ('train_r3gan', 'R3GAN', '49'),
            ('train_r3gan_adam', 'R3GAN + Adam (0 ; 0,99)', '1540'))
    fig, axes = plt.subplots(1, 5, figsize=(8.6, 2.45))
    for ax, (run, title, score) in zip(axes, runs):
        ax.imshow(plt.imread(f'{OUT}/noe_grid_{run}.png'), interpolation='nearest')
        ax.set_title(f'{title}\nFID {score}', fontsize=9.5, color=INK)
        ax.axis('off')
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f'{OUT}/noe_ablation.png', dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the baseline figures for the slides.')
    parser.add_argument('--weights', required=True, help='Dossier contenant G_ep50.pth, G_ep120.pth et G_ep200.pth.')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    evaluations = load_evaluations()
    plot_curves(evaluations)
    plot_grids(evaluations, args.weights)
    plot_classes(evaluations)
    plot_noe_ablation()
    print(f'Figures écrites dans {OUT}/')
