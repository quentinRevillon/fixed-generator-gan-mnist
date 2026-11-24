"""Figures de docs/training.md. Depuis la racine du repo : PYTHONPATH=. python figures/make_training_figures.py"""
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from model import Generator
from utils import load_model

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
RUNS = [('baseline_rerun', 'Baseline (BCE)', '#2a78d6'),
        ('train_rpgan_r1', 'RpGAN + R1', '#eb6834'),
        ('train_r3gan', 'R3GAN (RpGAN + R1 + R2)', '#1baf7a')]

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.edgecolor': GRID, 'font.size': 11})


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# 1. Exemples : vraies images + 3 modèles, même bruit z
z = torch.randn(64, 100, generator=torch.Generator().manual_seed(0))
real = datasets.MNIST(os.getenv('DATA', 'data'), train=False, transform=transforms.ToTensor())
panels = [('Vraies images (MNIST test)', torch.stack([real[i][0] for i in range(64)]))]
for name, label, _ in RUNS:
    G = load_model(Generator(784), f'runs/{name}/G.pth', 'cpu').eval()
    with torch.no_grad():
        panels.append((label, (G(z).view(-1, 1, 28, 28) + 1) / 2))
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, (title, imgs) in zip(axes, panels):
    grid = torchvision.utils.make_grid(imgs, nrow=8, padding=2, pad_value=1).permute(1, 2, 0).numpy()
    ax.imshow(grid, cmap='gray'); ax.axis('off'); ax.set_title(title, fontsize=13)
fig.suptitle('Même bruit z (seed 0) pour les trois générateurs — 50 époques', fontsize=12, color=INK2)
plt.tight_layout(); plt.savefig('figures/examples_same_z.png', dpi=110); plt.close()

# 2. FID au cours de l'entraînement (évalué toutes les 10 époques)
pat = re.compile(r'^\[(\d+)/\d+\].*FID=([0-9.]+) P=([0-9.]+) R=([0-9.]+)')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
# décalage vertical des étiquettes de fin pour éviter les chevauchements (RpGAN+R1 / R3GAN sont proches)
end_dy = {'baseline_rerun': (0, 0), 'train_rpgan_r1': (-7, 7), 'train_r3gan': (7, -7)}
for name, label, color in RUNS:
    rows = [tuple(map(float, m.groups())) for m in map(pat.match, open(f'logs/{name}.log')) if m]
    ep, fid, _, rec = map(np.array, zip(*rows))
    for ax, y, dy in ((axes[0], fid, end_dy[name][0]), (axes[1], rec, end_dy[name][1])):
        ax.plot(ep, y, color=color, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label)
        ax.annotate(f'{y[-1]:.2f}' if y[-1] < 1 else f'{y[-1]:.0f}', (ep[-1], y[-1]),
                    xytext=(8, dy), textcoords='offset points', va='center', color=INK2, fontsize=10)
axes[0].set_yscale('log'); axes[0].set_ylabel('FID (échelle log, ↓ meilleur)')
axes[0].set_title('FID au cours de l\'entraînement', loc='left')
axes[1].set_ylabel('Recall (↑ meilleur)'); axes[1].set_ylim(0, 1.0)
axes[1].axhline(0.87, color=INK2, linestyle='--', linewidth=1)
axes[1].text(10, 0.835, 'plafond : vraies vs vraies images = 0.87', color=INK2, fontsize=9)
axes[1].set_title('Recall (couverture) au cours de l\'entraînement', loc='left')
for ax in axes:
    style(ax); ax.set_xlabel('Époque'); ax.set_xticks([10, 20, 30, 40, 50]); ax.set_xlim(7, 56)
axes[0].legend(frameon=False, loc='upper right')
plt.tight_layout(); plt.savefig('figures/fid_recall_curves.png', dpi=110); plt.close()

# 3. Répartition des chiffres générés (classifieur du groupe, 10 000 images)
res = {r['run']: r for r in csv.DictReader(open('results.csv'))}
fig, ax = plt.subplots(figsize=(13, 4.4))
w = 0.26
for i, (name, label, color) in enumerate(RUNS):
    pct = np.array(list(map(int, res[name]['classes'].split('-')))) / 100
    ax.bar(np.arange(10) + (i - 1) * w, pct, width=w - 0.03, color=color, label=label)
ax.axhline(10, color=INK2, linestyle='--', linewidth=1)
ax.text(-0.45, 10.3, 'équilibre : 10 %', color=INK2, fontsize=9)
ax.set_xticks(range(10)); ax.set_xlabel('Chiffre prédit par le classifieur'); ax.set_ylabel('% des images générées')
ax.set_ylim(0, 21)
ax.set_title('Quels chiffres chaque modèle produit-il ?', loc='left')
style(ax); ax.legend(frameon=False, ncol=3, loc='upper right')
plt.tight_layout(); plt.savefig('figures/class_distribution.png', dpi=110); plt.close()
print('ok')
