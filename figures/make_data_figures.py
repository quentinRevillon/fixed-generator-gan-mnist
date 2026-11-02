"""Tableaux et figures de rapport-donnée.md. À lancer depuis la racine : python figures/make_data_figures.py [--real]

--real : mesures sur les vraies images seules (aucun GAN), sinon bilan des runs de configs/data/.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import torch
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, '.')
from evaluate import evaluate, mnist
from preprocessing import embed, gaussian_score, select_instances

OUT = 'figures/data'
DEVICE = torch.device('cpu')
SUBSETS = {'sel_75': (0.75, True, 'classifier'), 'sel_50': (0.5, True, 'classifier'),
           'sel_50_global': (0.5, False, 'classifier'), 'sel_50_pixels': (0.5, True, 'pixels')}
RUNS = {'data_ref': (60000, '#333333'), 'sel_75': (45000, '#1f77b4'), 'sel_50': (30000, '#ff7f0e'),
        'sel_50_global': (30000, '#d62728'), 'sel_50_pixels': (30000, '#9467bd'), 'noise_01': (60000, '#2ca02c'),
        'noise_03': (60000, '#17becf'), 'noise_01_const': (60000, '#8c564b'), 'sel_50_noise_01': (30000, '#e377c2')}


def real_subsets():
    """Ce que voit le GAN : chaque sous-ensemble comparé au MNIST test, comme une image générée."""
    x, y = mnist(train=True)
    with open(f'{OUT}/real_subsets.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'n', 'fid', 'precision', 'recall', 'classes_pct'])
        for name, (keep, per_class, space) in {'all': (1., True, 'classifier'), **SUBSETS}.items():
            idx = select_instances(y, embed(x, space, DEVICE), keep, per_class)
            rng = torch.Generator().manual_seed(0)  # même taille pour tous : les métriques en dépendent
            sub = idx[torch.randperm(len(idx), generator=rng)[:10000]]
            m = evaluate(x[sub], DEVICE)
            pct = 100 * torch.bincount(y[idx], minlength=10) / len(idx)
            row = [name, len(idx), f"{m['fid']:.1f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                   ' '.join(f'{p:.1f}' for p in pct)]
            w.writerow(row)
            print(row, flush=True)


def extremes():
    """Pour chaque chiffre : les 10 images les plus denses (gauche) et les 10 moins denses (droite)."""
    x, y = mnist(train=True)
    f, rows = embed(x, 'classifier', DEVICE), []
    for c in range(10):
        g = (y == c).nonzero().squeeze(1)
        order = g[gaussian_score(f[g]).argsort(descending=True)]
        rows.append(torch.cat([order[:10], order[-10:]]))
    grid = torchvision.utils.make_grid(x[torch.cat(rows)], nrow=20, padding=1, pad_value=0.5)
    torchvision.utils.save_image(grid, f'{OUT}/kept_vs_rejected.png')


def final_scores():
    """Dernière ligne de results.csv pour chaque run, regroupée par variante."""
    rows = {}
    with open('results.csv') as f:
        for r in csv.DictReader(f):
            rows[r['run']] = r
    out = defaultdict(list)
    for name in RUNS:
        for s in range(3):
            if f'{name}_s{s}' in rows:
                out[name].append(rows[f'{name}_s{s}'])
    return out


def write_summary(scores):
    with open(f'{OUT}/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'seeds', 'fid', 'fid_std', 'precision', 'precision_std', 'recall', 'recall_std',
                    'classes_pct', 'fid_values'])
        for name, rs in scores.items():
            vals = [np.array([float(r[k]) for r in rs]) for k in ('fid', 'precision', 'recall')]
            classes = np.mean([[int(c) for c in r['classes'].split('-')] for r in rs], 0) / 100
            row = [name, len(rs)]
            for v, d in zip(vals, (1, 3, 3)):
                row += [f'{v.mean():.{d}f}', f'{v.std():.{d}f}']
            row += [' '.join(f'{c:.1f}' for c in classes), ' '.join(f'{v:.1f}' for v in vals[0])]
            w.writerow(row)
            print(row)


def curves():
    """FID et recall en fonction du nombre de mises à jour, moyenne des seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (n_images, color) in RUNS.items():
        series = defaultdict(list)
        for run in glob.glob(f'runs/{name}_s[0-9]'):
            ea = EventAccumulator(run, size_guidance={'scalars': 0})
            ea.Reload()
            if 'eval/fid' not in ea.Tags()['scalars']:
                continue
            for tag in ('eval/fid', 'eval/recall'):
                for e in ea.Scalars(tag):
                    series[tag, e.step * n_images // 100].append(e.value)
        for ax, tag in zip(axes, ('eval/fid', 'eval/recall')):
            pts = sorted((u, np.mean(v)) for (t, u), v in series.items() if t == tag)
            if pts:
                ax.plot(*zip(*pts), color=color, label=name)
    axes[0].set(xlabel='mises à jour', ylabel='FID', yscale='log')
    axes[1].set(xlabel='mises à jour', ylabel='recall')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/curves.png', dpi=150)


def class_mix(scores):
    """Part de chaque chiffre dans les images générées, moyenne des seeds."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(scores)
    m = np.array([np.mean([[int(c) for c in r['classes'].split('-')] for r in scores[n]], 0) / 100 for n in names])
    im = ax.imshow(m, cmap='RdBu_r', vmin=0, vmax=20, aspect='auto')
    ax.set(xticks=range(10), yticks=range(len(names)), yticklabels=names, xlabel='chiffre prédit')
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, label='%')
    fig.tight_layout()
    fig.savefig(f'{OUT}/classes.png', dpi=150)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if '--real' in sys.argv:
        real_subsets()
        extremes()
    else:
        scores = final_scores()
        write_summary(scores)
        curves()
        class_mix(scores)
