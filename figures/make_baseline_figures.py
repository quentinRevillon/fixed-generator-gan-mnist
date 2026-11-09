"""Figures de baseline.md dans figures/baseline/, et chiffres à reporter dans le document.

Lit runs/baseline_200ep_prof/metrics.jsonl et trois générateurs (époques 50, 120 et 200).

Utilisation, depuis la racine du repo :
    python figures/make_baseline_figures.py --weights <dossier contenant G_ep50.pth, G_ep120.pth, G_ep200.pth>
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

from data import mnist_split
from evaluate import evaluate, generate, load_classifier, predict, to_unit_images
from model import Generator
from utils import load_model

OUT = 'figures/baseline'
RUN = 'runs/baseline_200ep_prof'
EPOCHS = (50, 120, 200)
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def load_evaluations():
    """Lignes de metrics.jsonl où les métriques ont été calculées (toutes les 2 époques)."""
    with open(f'{RUN}/metrics.jsonl') as f:
        return [r for r in map(json.loads, f) if 'fid' in r]


def imbalance(classes):
    """Part des images (%) qui devrait changer de chiffre pour atteindre 10 % partout."""
    shares = 100 * np.array(classes) / sum(classes)
    return float(np.abs(shares - 10).sum() / 2)


def plot_fid_panel(ax, rows, key, title, ticks):
    """FID par époque en échelle log, meilleure époque annotée sous la courbe."""
    epochs, values = [r['epoch'] for r in rows], [r[key] for r in rows]
    best = min(rows, key=lambda r: r[key])
    ax.plot(epochs, values, color=BLUE, linewidth=2)
    ax.scatter([best['epoch']], [best[key]], color=INK, s=36, zorder=3)
    ax.annotate(f"minimum : {best[key]:.1f}, époque {best['epoch']}", (best['epoch'], best[key]), xytext=(0, -9),
                textcoords='offset points', ha='right' if best['epoch'] > 150 else 'center', va='top', fontsize=9, color=INK)
    ax.set(yscale='log', ylabel='FID, échelle log', xlim=(0, 205), ylim=(best[key] / 1.8, max(values) * 1.2))
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.grid(axis='y', color=GRID)


def plot_pr_panel(ax, rows, precision, recall):
    """Précision et rappel par époque."""
    epochs = [r['epoch'] for r in rows]
    for key, label, color in ((precision, 'Précision', BLUE), (recall, 'Rappel', ORANGE)):
        ax.plot(epochs, [r[key] for r in rows], color=color, linewidth=2, label=label)
    ax.set(xlabel='Époque', ylabel='Valeur', xlim=(0, 205), ylim=(0, 1))
    ax.grid(axis='y', color=GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left', ncol=2)


def plot_curves(rows):
    """Les deux mesures côte à côte : FID en haut, précision et rappel en bas."""
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6), sharex=True)
    plot_fid_panel(axes[0, 0], rows, 'fid_prof', 'Inception et VGG (mesure du classement)', (40, 60, 100, 150, 250))
    plot_fid_panel(axes[0, 1], rows, 'fid', 'Petit CNN MNIST', (50, 100, 200, 500, 1000, 2000))
    plot_pr_panel(axes[1, 0], rows, 'precision_prof', 'recall_prof')
    plot_pr_panel(axes[1, 1], rows, 'precision', 'recall')
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_curves.png', dpi=200)


@torch.no_grad()
def plot_grids(rows, weights):
    """Mêmes z pour les trois générateurs : seule l'époque change."""
    z = torch.randn(36, 100, generator=torch.Generator().manual_seed(0))
    by_epoch = {r['epoch']: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.5))
    for ax, epoch in zip(axes, EPOCHS):
        G = load_model(Generator(784), os.path.join(weights, f'G_ep{epoch}.pth'), 'cpu').eval()
        grid = torchvision.utils.make_grid(to_unit_images(G(z)), nrow=6, padding=2, pad_value=1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f"Époque {epoch}\nFID Inception {by_epoch[epoch]['fid_prof']:.0f}, CNN {by_epoch[epoch]['fid']:.0f}",
                     fontsize=10, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_grids.png', dpi=200)


def plot_classes(rows):
    """Part de chaque chiffre aux époques 50, 120 et 200."""
    by_epoch = {r['epoch']: 100 * np.array(r['classes']) / sum(r['classes']) for r in rows}
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    for i, (epoch, color, hatch) in enumerate(zip(EPOCHS, (AQUA, BLUE, ORANGE), ('', '', '//'))):
        ax.bar(np.arange(10) + (i - 1) * 0.27, by_epoch[epoch], 0.25, color=color, hatch=hatch, edgecolor=SURFACE,
               linewidth=0.5, label=f'Époque {epoch}')
    ax.axhline(10, color=MUTED, linestyle=':', linewidth=1.2)
    ax.annotate('réel : ≈ 10 %', (9.55, 10), xytext=(0, 3), textcoords='offset points', fontsize=9, color=MUTED)
    ax.set(xticks=range(10), xlabel='Chiffre prédit par le petit CNN', ylabel='% des images', xlim=(-0.7, 11), ylim=(0, 29))
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper right', labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_classes.png', dpi=200)
    return by_epoch


def plot_imbalance(rows):
    """Déséquilibre des chiffres au cours de l'entraînement."""
    late = [r for r in rows if r['epoch'] >= 60]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.plot([r['epoch'] for r in late], [imbalance(r['classes']) for r in late], color=BLUE, linewidth=2)
    ax.set(xlabel='Époque', ylabel='Écart (%)', xlim=(58, 202), ylim=(0, None))
    ax.grid(axis='y', color=GRID)
    fig.tight_layout()
    fig.savefig(f'{OUT}/baseline_imbalance.png', dpi=200)


def rebalanced_images(images, preds, labels):
    """Sous-échantillon qui a exactement la répartition des chiffres de `labels`."""
    wanted = torch.bincount(labels, minlength=10).tolist()
    kept = [images[preds == digit][:count] for digit, count in enumerate(wanted)]
    return torch.cat(kept)


def rebalanced_fid(G, device, n_pool=100_000):
    """(FID petit CNN avec 10 000 images telles quelles, FID après remise des chiffres à la répartition de val)."""
    images = generate(G, n_pool, device)
    preds = predict(load_classifier(device), images, device)[1]
    balanced = rebalanced_images(images, preds, mnist_split('val')[1])
    assert len(balanced) == 10_000
    return evaluate(images[:10_000], device)['fid'], evaluate(balanced, device)['fid']


def print_numbers(rows, by_epoch, weights):
    """Chiffres à reporter dans le document."""
    for r in rows:
        if r['epoch'] in (50, 80, 106, 120, 140, 160, 170, 182, 194, 200):
            print(f"{r['epoch']} | INC {r['fid_prof']:.1f} P {r['precision_prof']:.3f} R {r['recall_prof']:.3f} | "
                  f"CNN {r['fid']:.1f} P {r['precision']:.3f} R {r['recall']:.3f} | ecart {imbalance(r['classes']):.1f}")
    for epoch in EPOCHS:
        print(epoch, np.round(by_epoch[epoch], 1).tolist())
    late = [r for r in rows if r['epoch'] >= 100]
    gaps = [imbalance(r['classes']) for r in late]
    print('correlation, epoques >= 100, ecart/FID CNN', round(np.corrcoef(gaps, [r['fid'] for r in late])[0, 1], 2),
          'ecart/FID Inception', round(np.corrcoef(gaps, [r['fid_prof'] for r in late])[0, 1], 2))
    print('variation du FID Inception entre 2 mesures, epoques >= 160 (moyenne des ecarts absolus)',
          round(np.abs(np.diff([r['fid_prof'] for r in rows if r['epoch'] >= 160])).mean(), 1))
    for epoch in (120, 200):
        G = load_model(Generator(784), os.path.join(weights, f'G_ep{epoch}.pth'), 'cpu').eval()
        print('FID CNN tel quel, puis chiffres remis en proportion : epoque', epoch, np.round(rebalanced_fid(G, 'cpu'), 1).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the baseline figures.')
    parser.add_argument('--weights', required=True, help='Dossier contenant G_ep50.pth, G_ep120.pth et G_ep200.pth.')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    evaluations = load_evaluations()
    plot_curves(evaluations)
    plot_grids(evaluations, args.weights)
    shares = plot_classes(evaluations)
    plot_imbalance(evaluations)
    print_numbers(evaluations, shares, args.weights)
