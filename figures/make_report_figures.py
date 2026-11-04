"""Figures de rapport-donnée-illustré.md. À lancer depuis la racine, après make_data_figures.py :
python figures/make_report_figures.py
"""
import csv
import glob
import io
import os
import sys

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, '.')
sys.path.insert(0, 'figures')
from evaluate import mnist
from make_data_figures import BUDGET, DEVICE, RUNS, load_runs
from model import Discriminator
from preprocessing import embed, select_instances

OUT = 'figures/report'
SHOWN = ['data_ref', 'sel_75', 'sel_50', 'sel_50_global', 'noise_01', 'noise_03']
REAL = {'data_ref': 'all', 'sel_75': 'sel_75', 'sel_50': 'sel_50', 'sel_50_global': 'sel_50_global',
        'sel_50_pixels': 'sel_50_pixels', 'sel_50_noise_01': 'sel_50'}


def epoch_at(name, updates):
    return round(updates * 100 / RUNS[name][0])


def samples(run, epoch):
    """Grille 8×8 enregistrée par train.py à cette époque (même z fixe pour tous les runs)."""
    ea = EventAccumulator(run, size_guidance={'images': 0})
    ea.Reload()
    im = {e.step: e for e in ea.Images('samples')}[epoch]
    return np.array(Image.open(io.BytesIO(im.encoded_image_string)).convert('L'))


def first_ok(runs, name):
    return next(i for i, r in enumerate(runs[name]) if not r['collapsed'])


def scores(runs):
    """FID, precision et recall à 60 000 mises à jour : un point par run, barre = moyenne, losange = vraies images."""
    real = {r['subset']: r for r in csv.DictReader(open('figures/data/real_subsets.csv'))}
    names = list(runs)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ('fid', 'precision', 'recall')):
        for i, name in enumerate(names):
            v = [r['curves'][key][r['at']] for r in runs[name] if not r['collapsed']]
            ax.bar(i, np.mean(v), color=RUNS[name][1], alpha=0.6)
            ax.scatter([i] * len(v), v, color='black', s=10, zorder=3)
            if name in REAL:
                ax.scatter(i, float(real[REAL[name]][key]), marker='D', color='white', edgecolor='black', s=35, zorder=4)
        ax.set_xticks(range(len(names)), names, rotation=60, ha='right', fontsize=8)
        ax.set_title(key + (' (bas = mieux)' if key == 'fid' else ' (haut = mieux)'), fontsize=10)
    axes[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(f'{OUT}/scores.png', dpi=150)


def pr_scatter(runs):
    """Compromis qualité / diversité : chaque run à 60 000 mises à jour, et les vraies images sélectionnées."""
    real = {r['subset']: r for r in csv.DictReader(open('figures/data/real_subsets.csv'))}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rs in runs.items():
        ok = [r for r in rs if not r['collapsed']]
        ax.scatter([r['curves']['recall'][r['at']] for r in ok], [r['curves']['precision'][r['at']] for r in ok],
                   color=RUNS[name][1], label=name, s=30)
    for sub, r in real.items():
        ax.scatter(float(r['recall']), float(r['precision']), marker='*', s=160, color='gold', edgecolor='black')
        ax.annotate(f'réel {sub}', (float(r['recall']), float(r['precision'])), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    ax.set(xlabel='recall (diversité)', ylabel='precision (qualité)')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(f'{OUT}/precision_recall.png', dpi=150)


def curves(runs):
    """FID moyen ± écart-type selon le nombre de mises à jour, sélection et bruit séparés."""
    groups = [['data_ref', 'sel_75', 'sel_50', 'sel_50_global'], ['data_ref', 'noise_01', 'noise_03', 'noise_01_const']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, group, title in zip(axes, groups, ('Instance Selection', 'Instance Noise')):
        for name in group:
            ok = [r for r in runs[name] if not r['collapsed']]
            xs = np.array([s * RUNS[name][0] // 100 for s in ok[0]['steps']])
            v = np.array([[r['curves']['fid'][s] for s in r['steps']] for r in ok])
            ax.plot(xs, v.mean(0), color=RUNS[name][1], label=f'{name} (n={len(ok)})')
            ax.fill_between(xs, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=RUNS[name][1], alpha=0.15)
        ax.axvline(BUDGET, color='#999999', lw=0.8, ls='--')
        ax.set(title=title, xlabel='mises à jour', yscale='log')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('FID')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fid_curves.png', dpi=150)


def examples(runs):
    """Images générées à 60 000 mises à jour, même z pour toutes les variantes."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, name in zip(axes.flat, SHOWN):
        s = first_ok(runs, name)
        r = runs[name][s]
        ax.imshow(samples(f'runs/{name}_s{s}', epoch_at(name, BUDGET)), cmap='gray')
        ax.set_title(f"{name} (seed {s}) : FID {r['curves']['fid'][r['at']]:.0f}", fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/examples_60k.png', dpi=130)


def evolution(runs):
    """Même z au fil de l'entraînement : la référence se dégrade après 60 000 mises à jour."""
    steps = [12000, 36000, 60000, 120000]
    names = ['data_ref', 'sel_50']
    fig, axes = plt.subplots(len(names), len(steps), figsize=(12, 6.4))
    for row, name in zip(axes, names):
        s = first_ok(runs, name)
        fids = runs[name][s]['curves']['fid']
        for ax, u in zip(row, steps):
            e = epoch_at(name, u)
            ax.imshow(samples(f'runs/{name}_s{s}', e), cmap='gray')
            near = min(fids, key=lambda k: abs(k - e))
            ax.set_title(f'{name}, {u // 1000}k mises à jour (FID {fids[near]:.0f})', fontsize=8)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/evolution.png', dpi=130)


def collapse(runs):
    """D(x) des runs de référence : les runs effondrés saturent dès les premières époques ; exemple d'image."""
    fig, (ax, im) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1.6, 1]})
    bad = None
    for s, r in enumerate(runs['data_ref']):
        ea = EventAccumulator(f'runs/data_ref_s{s}', size_guidance={'scalars': 0})
        ea.Reload()
        e = ea.Scalars('D(x)')[:40]
        ax.plot([x.step for x in e], [x.value for x in e], color='#d62728' if r['collapsed'] else '#333333',
                alpha=0.8, label='effondré' if r['collapsed'] else 'normal')
        if r['collapsed'] and bad is None:
            bad = s
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.set(xlabel='époque', ylabel='D(x) moyen', title='data_ref, 8 seeds : 40 premières époques')
    im.imshow(samples(f'runs/data_ref_s{bad}', 100), cmap='gray')
    im.set_title(f'data_ref seed {bad}, 60k mises à jour', fontsize=9)
    im.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/collapse.png', dpi=140)


def class_bars(runs):
    """Part de chaque chiffre : données d'entraînement, générées à 60k et à la fin."""
    real = 9.9, 11.2, 9.9, 10.2, 9.7, 9.0, 9.9, 10.4, 9.8, 9.9
    final = {r['run']: [int(c) for c in r['classes'].split('-')]
             for r in csv.DictReader(open('figures/data/results_juliet.csv'))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, name in zip(axes, ('data_ref', 'sel_50')):
        ok = [(s, r) for s, r in enumerate(runs[name]) if not r['collapsed']]
        mid = np.mean([r['classes'] for _, r in ok], 0)
        end = np.mean([100 * np.array(final[f'{name}_s{s}']) / sum(final[f'{name}_s{s}']) for s, _ in ok], 0)
        x = np.arange(10)
        ax.bar(x - 0.27, real, 0.27, color='#bbbbbb', label='données vues (équilibrées)')
        ax.bar(x, mid, 0.27, color=RUNS[name][1], label='générées à 60k')
        ax.bar(x + 0.27, end, 0.27, color=RUNS[name][1], alpha=0.45, label='générées à la fin')
        ax.set(title=name, xticks=x, xlabel='chiffre')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('% des images')
    fig.tight_layout()
    fig.savefig(f'{OUT}/class_bars.png', dpi=140)


@torch.no_grad()
def d_scores():
    """Score du D final de sel_50 seed 0 : images vues, images rejetées (jamais vues) et MNIST test."""
    x, y = mnist(train=True)
    xt = mnist(train=False)[0]
    seen = select_instances(y, embed(x, 'classifier', DEVICE), 0.5)
    mask = torch.ones(len(y), dtype=torch.bool)
    mask[seen] = False
    D = Discriminator(784)
    D.load_state_dict(torch.load('runs/sel_50_s0/D.pth', map_location='cpu'))
    D.eval()
    score = lambda imgs: D(imgs[:10000].flatten(1) * 2 - 1).squeeze(1).numpy()
    g = torch.Generator().manual_seed(0)
    pick = lambda idx: x[idx[torch.randperm(len(idx), generator=g)]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for imgs, label, color in ((pick(seen), 'vues à l\'entraînement', '#ff7f0e'),
                               (pick(mask.nonzero().squeeze(1)), 'train rejetées (jamais vues)', '#1f77b4'),
                               (xt, 'MNIST test', '#333333')):
        v = score(imgs)
        ax.hist(v, bins=50, range=(0, 1), histtype='step', lw=1.6, color=color, label=f'{label} : moyenne {v.mean():.2f}')
    ax.set(xlabel='D(x) : probabilité « réelle »', ylabel='images', title='sel_50 seed 0, D en fin d\'entraînement')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{OUT}/d_scores.png', dpi=140)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()
    for f in (scores, pr_scatter, curves, examples, evolution, collapse, class_bars):
        f(runs)
        print(f.__name__, flush=True)
    d_scores()
