"""Figures de la campagne 100 époques + EMA (training-r3gan.md, section 6).
Depuis la racine du repo : DATA=<chemin MNIST> PYTHONPATH=. python figures/make_campaign_figures.py"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from model import Generator
from utils import load_model

SURFACE, INK, INK2, GRID, BLUE = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.edgecolor': GRID, 'font.size': 11})
CONFIGS = [('Baseline\n(BCE)', 'ema100_baseline_s{}'), ('RpGAN + R1\nγ = 1', 'ema100_rpgan_r1_s{}'),
           ('R3GAN\nγ = 1', 'ema100_r3gan_s{}'), ('RpGAN + R1\nγ = 10', 'ema100_rpgan_r1_g10_s{}')]
COLLAPSE = 1000  # au-delà, le run est effondré : affiché à part

# 1. FID / precision / recall par configuration, un point par seed
res = {r['run']: r for r in csv.DictReader(open('results.csv'))}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, key, title in zip(axes, ('fid', 'precision', 'recall'), ('FID (↓)', 'Precision (↑)', 'Recall (↑)')):
    for i, (label, pat) in enumerate(CONFIGS):
        v = np.array([float(res[pat.format(s)][key]) for s in range(3)])
        ok = np.array([float(res[pat.format(s)]['fid']) < COLLAPSE for s in range(3)])
        ax.scatter(np.full(ok.sum(), i), v[ok], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.hlines(v[ok].mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.annotate(f'{v[ok].mean():.2f}' if key != 'fid' else f'{v[ok].mean():.1f}', (i + 0.28, v[ok].mean()),
                    va='center', color=INK2, fontsize=10)
        if (~ok).any() and key == 'fid':
            ax.annotate('1 seed sur 3\neffondrée\n(FID 2770)', (i, 75), ha='center', color=INK2, fontsize=9)
    ax.set_xticks(range(len(CONFIGS))); ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=10)
    ax.set_title(title, loc='left'); ax.set_xlim(-0.5, len(CONFIGS) - 0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8); ax.set_axisbelow(True)
axes[0].set_ylim(0, 90)
for ax in axes[1:]:  # même échelle pour P et R, avec le plafond « vraies vs vraies images »
    ax.set_ylim(0.6, 0.9)
    ax.axhline(0.87, color=INK2, linestyle='--', linewidth=1)
    ax.text(-0.45, 0.876, 'plafond : 0.87', color=INK2, fontsize=9)
fig.suptitle('100 époques, EMA 0.999 — un point par seed, trait = moyenne (seeds non effondrées). '
             'γ = 0.1 : diverge sur les 3 seeds.', fontsize=11, color=INK2)
plt.tight_layout(); plt.savefig('figures/campaign_seeds.png', dpi=110); plt.close()

# 2. Exemples : vraies images, baseline, meilleur RpGAN + R1 (même bruit z)
z = torch.randn(64, 100, generator=torch.Generator().manual_seed(0))
real = datasets.MNIST(os.getenv('DATA', 'data'), train=False, transform=transforms.ToTensor())
panels = [('Vraies images (MNIST test)', torch.stack([real[i][0] for i in range(64)]))]
for title, run in (('Baseline — 100 ép. + EMA (seed 1)', 'ema100_baseline_s1'),
                   ('RpGAN + R1 — 100 ép. + EMA (seed 1)', 'ema100_rpgan_r1_s1')):
    G = load_model(Generator(784), f'runs/{run}/G.pth', 'cpu').eval()
    with torch.no_grad():
        imgs = (G(z).view(-1, 1, 28, 28) + 1) / 2
    r = res[run]
    panels.append((f"{title}\nFID {float(r['fid']):.1f} | P {float(r['precision']):.2f} | R {float(r['recall']):.2f}", imgs))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
for ax, (title, imgs) in zip(axes, panels):
    grid = torchvision.utils.make_grid(imgs, nrow=8, padding=2, pad_value=1).permute(1, 2, 0).numpy()
    ax.imshow(grid, cmap='gray'); ax.axis('off'); ax.set_title(title, fontsize=12)
plt.tight_layout(); plt.savefig('figures/examples_best.png', dpi=110); plt.close()
print('ok')
