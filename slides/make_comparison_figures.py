"""Figures de slides/comparaison.tex (dans slides/figures/).

Lit runs/baseline_200ep_prof/metrics.jsonl, les générateurs et, s'il existe, results/comparison_test.csv
(produit par scripts/measure_runs.py).

Utilisation, depuis la racine du repo :
    python slides/make_comparison_figures.py --weights <dossier contenant les G de Noé et de Nicodème>
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from evaluate import to_unit_images
from model import Discriminator, Generator
from sampling import sample
from utils import load_model

OUT = 'slides/figures'
RESULTS = 'results/comparison_test.csv'
RUN = 'runs/baseline_200ep_prof'
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
# Nom dans le CSV -> (étiquette de la slide, couleur, marque pleine). Les références de Noé et Nicodème sont creuses.
CONFIGS = {'Baseline commune': ('Baseline commune', INK, True),
           'DRS percentile 80': ('Inférence : DRS', BLUE, True),
           'Reference de Noe (50 epoques)': ('Référence de Noé, 50 époques', ORANGE, False),
           'RpGAN + R1 (50 epoques)': ('Entraînement : RpGAN + R1, 50 époques', ORANGE, True),
           'R3GAN (50 epoques)': ('Entraînement : R3GAN, 50 époques', ORANGE, True),
           'Reference de Nicodeme': ('Référence de Nicodème', AQUA, False),
           'Bruit 0.1': ('Données : bruit 0,1', AQUA, True),
           'Bruit 0.3': ('Données : bruit 0,3', AQUA, True),
           'Selection 75 %': ('Données : sélection 75 %', AQUA, True),
           'Selection 50 %': ('Données : sélection 50 %', AQUA, True)}
GRIDS = (('Baseline', None), ('Bruit 0,1', 'nico_noise_01_s1.pth'), ('Sélection 50 %', 'nico_sel_50_s0.pth'),
         ('R3GAN, 50 époques', 'noe_train_r3gan.pth'), ('DRS', 'drs'))
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def plot_two_measures():
    """FID de la baseline selon les deux mesures, côte à côte, chaque minimum annoté."""
    with open(f'{RUN}/metrics.jsonl') as f:
        rows = [r for r in map(json.loads, f) if 'fid_prof' in r]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    panels = (('fid_prof', 'Inception (mesure du classement)', (40, 60, 100, 150, 250)),
              ('fid', 'Petit CNN MNIST', (50, 100, 200, 500, 1000, 2000)))
    for ax, (key, title, ticks) in zip(axes, panels):
        best = min(rows, key=lambda r: r[key])
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=INK, linewidth=2)
        ax.scatter([best['epoch']], [best[key]], color=INK, s=45, zorder=3)
        ax.annotate(f"minimum : époque {best['epoch']}", (best['epoch'], best[key]), xytext=(0, -11), textcoords='offset points',
                    ha='right' if best['epoch'] > 150 else 'center', va='top', color=INK)
        ax.set(yscale='log', xlabel='Époque', ylabel='FID', xlim=(0, 205), ylim=(best[key] / 1.8, max(r[key] for r in rows) * 1.2))
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_title(title, loc='left', fontsize=13, color=INK)
        ax.grid(axis='y', color=GRID)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_two_measures.png', dpi=200)


def plot_drs_gain():
    """FID et précision de la baseline et du DRS sur l'époque 200, avec et sans KeepTraining, percentile 80."""
    with open('results/results_prof.csv', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['run'] == RUN and r['reference'] == 'val']
    configs = (('Sans tri', 'best', 'plain', MUTED), ('DRS', 'best', 'drs_azadi', BLUE), ('KeepTraining + DRS', 'best_kt', 'drs_azadi', ORANGE))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
    for ax, (key, title, fmt) in zip(axes, (('fid', 'FID (plus bas = mieux)', '{:.1f}'), ('precision', 'Précision', '{:.3f}'), ('recall', 'Rappel', '{:.3f}'))):
        values = [np.mean([float(r[key]) for r in rows if r['weights'] == w and r['method'] == m
                           and (m == 'plain' or json.loads(r['params'])['percentile'] == 80)]) for _, w, m, _ in configs]
        ax.barh([c[0] for c in configs], values, color=[c[3] for c in configs], height=0.65)
        for y, value in enumerate(values):
            ax.annotate(fmt.format(value).replace('.', ','), (value, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
        ax.set(xlim=(0, max(values) * 1.3))
        ax.set_title(title, loc='left', fontsize=13, color=INK)
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_drs_gain.png', dpi=200)


def plot_data_gain():
    """Résultats de Nicodème à 60 000 mises à jour (figures/data/summary.csv de sa branche, copié dans slides/figures)."""
    with open(f'{OUT}/nico_summary.csv', newline='') as f:
        rows = {r['variant']: r for r in csv.DictReader(f)}
    configs = (('Référence', 'data_ref', MUTED), ('Bruit 0,1', 'noise_01', AQUA), ('Bruit 0,3', 'noise_03', AQUA),
               ('Sélection 75 %', 'sel_75', BLUE), ('Sélection 50 %', 'sel_50', BLUE))
    panels = (('fid_60k', 'FID (plus bas = mieux)', '{:.0f}'), ('precision_60k', 'Précision', '{:.2f}'), ('recall_60k', 'Rappel', '{:.2f}'))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, (key, title, fmt) in zip(axes, panels):
        values = [float(rows[v][key]) for _, v, _ in configs]
        ax.barh([c[0] for c in configs], values, color=[c[2] for c in configs], height=0.65)
        for y, value in enumerate(values):
            ax.annotate(fmt.format(value).replace('.', ','), (value, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
        ax.set(xlim=(0, max(values) * 1.3))
        ax.set_title(title, loc='left', fontsize=13, color=INK)
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_data_gain.png', dpi=200)


def load_comparison():
    """{nom : agrégats} : moyenne et écart-type de FID, précision et rappel, écart à 10 % par chiffre."""
    with open(RESULTS, newline='') as f:
        groups = defaultdict(list)
        for row in csv.DictReader(f):
            groups[row['name']].append(row)
    out = {}
    for name, rows in groups.items():
        agg = {k: (np.mean([float(r[k]) for r in rows]), np.std([float(r[k]) for r in rows]))
               for k in ('fid', 'precision', 'recall', 'acceptance_rate')}
        counts = np.array([[int(c) for c in r['classes'].split('-')] for r in rows]).sum(0)
        agg['shares'] = 100 * counts / counts.sum()
        agg['imbalance'] = float(np.abs(agg['shares'] - 10).sum() / 2)
        out[name] = agg
    return out


def marker_style(color, filled):
    """Marque pleine, ou creuse pour une référence propre à un volet."""
    return {'color': color, 'facecolors': color if filled else SURFACE, 'edgecolors': color, 's': 90, 'linewidths': 2, 'zorder': 3}


def plot_fid(results):
    """FID de chaque configuration, valeurs écrites, baseline commune en trait vertical."""
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    names = [n for n in CONFIGS if n in results]
    ax.axvline(results['Baseline commune']['fid'][0], color=INK, linestyle=':', linewidth=1.2)
    for y, name in enumerate(names):
        label, color, filled = CONFIGS[name]
        value = results[name]['fid'][0]
        ax.scatter([value], [y], **{k: v for k, v in marker_style(color, filled).items() if k != 'color'})
        ax.annotate(f'{value:.1f}'.replace('.', ','), (value, y), xytext=(10, 0), textcoords='offset points', va='center', color=INK)
    ax.set_yticks(range(len(names)), [CONFIGS[n][0] for n in names])
    ax.invert_yaxis()
    ax.set(xlabel='FID Inception, contre MNIST test (plus bas = mieux)', xscale='log')
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_fid.png', dpi=200)


def plot_precision_recall(results):
    """Précision et rappel : une marque par configuration, avec son nom."""
    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    for name in CONFIGS:
        if name not in results:
            continue
        label, color, filled = CONFIGS[name]
        r, p = results[name]['recall'][0], results[name]['precision'][0]
        ax.scatter([r], [p], **{k: v for k, v in marker_style(color, filled).items() if k != 'color'})
        ax.annotate(label.split(' : ')[-1], (r, p), xytext=(8, 5), textcoords='offset points', fontsize=10, color=INK)
    ax.set(xlabel='Rappel VGG (diversité)', ylabel='Précision VGG (qualité)')
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_pr.png', dpi=200)


def plot_balance(results):
    """Écart à 10 % par chiffre de chaque configuration, axe partant de 0."""
    names = [n for n in CONFIGS if n in results]
    values = [results[n]['imbalance'] for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.barh([CONFIGS[n][0] for n in names], values, color=[CONFIGS[n][1] for n in names], height=0.68)
    for y, value in enumerate(values):
        ax.annotate(f'{value:.1f} %'.replace('.', ','), (value, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
    ax.invert_yaxis()
    ax.set(xlabel='Part des images à changer de chiffre pour atteindre 10 % partout', xlim=(0, max(values) * 1.15))
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_balance.png', dpi=200)


@torch.no_grad()
def images_of(g_file, weights, z):
    """36 images de la colonne : G(z) pour un générateur, images acceptées pour le DRS."""
    G = load_model(Generator(784), os.path.join(weights, g_file) if g_file not in (None, 'drs') else f'{RUN}/G_best.pth', 'cpu').eval()
    if g_file != 'drs':
        return G(z)
    D = load_model(Discriminator(784), f'{RUN}/D_best.pth', 'cpu').eval()
    return sample(G, D, len(z), {'method': 'drs_azadi', 'percentile': 80, 'n_burnin': 10_000}, seed=0, batch_size=1000)[0]


def plot_samples(weights):
    """Une grille par approche : mêmes z sauf pour le DRS, qui garde une partie des images proposées."""
    z = torch.randn(36, 100, generator=torch.Generator().manual_seed(0))
    fig, axes = plt.subplots(1, len(GRIDS), figsize=(13, 3.1))
    for ax, (title, g_file) in zip(axes, GRIDS):
        grid = torchvision.utils.make_grid(to_unit_images(images_of(g_file, weights, z)), nrow=6, padding=2, pad_value=1)
        ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
        ax.set_title(title, fontsize=13, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_samples.png', dpi=200)


def print_table(results):
    """Chiffres à reporter dans les slides."""
    for name in CONFIGS:
        if name in results:
            a = results[name]
            print(f"{name}: FID {a['fid'][0]:.1f}±{a['fid'][1]:.1f} P {a['precision'][0]:.3f} R {a['recall'][0]:.3f} "
                  f"acc {a['acceptance_rate'][0]:.3f} ecart {a['imbalance']:.1f} parts {np.round(a['shares'], 1).tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the comparison slides figures.')
    parser.add_argument('--weights', required=True, help='Dossier contenant les générateurs de Noé et de Nicodème.')
    args = parser.parse_args()
    plot_two_measures()
    plot_data_gain()
    plot_drs_gain()
    plot_samples(args.weights)
    if os.path.exists(RESULTS):
        results = load_comparison()
        plot_fid(results)
        plot_precision_recall(results)
        plot_balance(results)
        print_table(results)
