"""Figures de slides/comparaison.tex (dans slides/figures/), avec des textes en anglais.

Lit runs/baseline_200ep_prof/, figures/data/kept_vs_rejected.png (Nicodème), les générateurs de Noé et, s'il existe,
results/comparison_local.csv (campagne commune de Noé, scripts/compare_local.py).

Utilisation, depuis la racine du repo :
    python slides/make_comparison_figures.py --weights <dossier>
Le dossier contient G_ep50.pth, G_ep120.pth, G_ep200.pth, les noe_*.pth, nico_*.pth et cmploc_*.pth (poids de leurs branches).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from data import mnist_split
from evaluate import to_unit_images
from model import Discriminator, Generator
from sampling import calibration_pool, drs_f_hat, propose, sample
from utils import load_model

OUT = 'slides/figures'
RESULTS = 'results/comparison_local.csv'
RUN = 'runs/baseline_200ep_prof'
EPOCHS = (50, 120, 200)
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
LIGHT_BLUE = '#a9cbef'
plt.rcParams.update({'font.size': 15, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def load_evaluations():
    """Lignes de metrics.jsonl de la baseline où les métriques ont été calculées, par époque."""
    with open(f'{RUN}/metrics.jsonl') as f:
        return {r['epoch']: r for r in map(json.loads, f) if 'fid_prof' in r}


def image_grid(ax, x, title, nrow=6):
    """Affiche des images (n, 784) dans [-1, 1] en grille, avec un titre."""
    grid = torchvision.utils.make_grid(to_unit_images(x), nrow=nrow, padding=2, pad_value=1)
    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
    ax.set_title(title, fontsize=14, color=INK)
    ax.axis('off')


def fixed_noise(n=36):
    """Le même bruit z pour toutes les grilles."""
    return torch.randn(n, 100, generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def plot_epochs(rows, weights):
    """Mêmes z aux époques 50, 120 et 200, FID Inception en titre."""
    epochs = (50, 106, 200) if os.path.exists(os.path.join(weights, 'G_ep106.pth')) else EPOCHS
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, epoch in zip(axes, epochs):
        G = load_model(Generator(784), os.path.join(weights, f'G_ep{epoch}.pth'), 'cpu').eval()
        image_grid(ax, G(fixed_noise()), f"Epoch {epoch}\nInception FID {rows[epoch]['fid_prof']:.0f}, CNN FID {rows[epoch]['fid']:.0f}")
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_epochs.png', dpi=200)


def real_shares():
    """Part de chaque chiffre (%) dans le split d'entraînement, d'après les vraies étiquettes."""
    labels = mnist_split('train')[1]
    return (100 * labels.bincount(minlength=10) / len(labels)).numpy()


def digit_panel(ax, shares, title, real=None):
    """Barres de la part de chaque chiffre ; en option, repères orange à la vraie distribution."""
    ax.bar(range(10), shares, color=INK if real is not None else MUTED, width=0.7)
    if real is not None:
        ax.scatter(range(10), real, marker='_', s=420, linewidths=3, color=ORANGE, zorder=3, label='Real data')
        ax.legend(frameon=False, fontsize=12, labelcolor=MUTED, loc='upper right')
    for digit in (shares.argmax(), shares.argmin()):
        ax.annotate(f'{shares[digit]:.0f}%', (digit, shares[digit]), xytext=(0, 4), textcoords='offset points', ha='center', color=INK, fontsize=13)
    ax.set(xticks=range(10), xlabel='Digit', ylim=(0, 30))
    ax.set_title(title, loc='left', fontsize=14, color=INK)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)


def plot_digits(rows):
    """Vraie distribution, puis part de chaque chiffre au minimum de chaque FID. Renvoie les parts par époque."""
    real = real_shares()
    panels = ((min(rows, key=lambda e: rows[e]['fid']), 'best CNN FID'), (min(rows, key=lambda e: rows[e]['fid_prof']), 'best Inception FID'))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    digit_panel(axes[0], real, 'Real data')
    out = {'real': real}
    for ax, (epoch, name) in zip(axes[1:], panels):
        out[epoch] = 100 * np.array(rows[epoch]['classes']) / sum(rows[epoch]['classes'])
        digit_panel(ax, out[epoch], f'Epoch {epoch}: {name}', real)
    axes[0].set_ylabel('% of images')
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_digits.png', dpi=200)
    return out


def plot_selection():
    """La figure de Nicodème coupée en deux : images gardées et images retirées, chiffre par chiffre."""
    image = plt.imread('figures/data/kept_vs_rejected.png')
    middle = image.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3))
    for ax, part, title in zip(axes, (image[:, :middle], image[:, middle:]), ('Kept: highest s(x)', 'Removed: lowest s(x)')):
        ax.imshow(part, interpolation='nearest')
        ax.set_title(title, fontsize=16, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_selection.png', dpi=200)


@torch.no_grad()
def plot_data_samples(weights):
    """Même bruit : images générées par les modèles finaux de Nicodème, référence, bruit et sélection."""
    runs = (('Reference', 'nico_data_ref_s0.pth'), ('Noise 0.1', 'nico_noise_01_s1.pth'),
            ('Selection 75%', 'nico_sel_75_s0.pth'), ('Selection 50%', 'nico_sel_50_s0.pth'))
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 3.5))
    for ax, (title, name) in zip(axes, runs):
        G = load_model(Generator(784), os.path.join(weights, name), 'cpu').eval()
        image_grid(ax, G(fixed_noise()), title)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_data_samples.png', dpi=200)


@torch.no_grad()
def plot_loss_effect(weights):
    """Même bruit, 100 époques avec EMA : perte BCE contre RpGAN + R1 (campagne finale de Noé)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
    for ax, name, title in zip(axes, ('noe_ema100_baseline_s0.pth', 'noe_ema100_rpgan_r1_s1.pth'), ('BCE loss', 'RpGAN + R1')):
        G = load_model(Generator(784), os.path.join(weights, name), 'cpu').eval()
        image_grid(ax, G(fixed_noise()), title)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_loss_effect.png', dpi=200)


@torch.no_grad()
def plot_kept_rejected(percentile=80, n=36):
    """Mêmes propositions de G à l'époque 200, séparées par la décision du DRS."""
    G = load_model(Generator(784), f'{RUN}/G_best.pth', 'cpu').eval()
    D = load_model(Discriminator(784), f'{RUN}/D_best.pth', 'cpu').eval()
    rng = torch.Generator().manual_seed(0)
    log_M = calibration_pool(G, D, rng, 10_000, 2000).max().item()
    x, log_r = propose(G, D, 2000, rng, 2000)
    f_hat = drs_f_hat(log_r, max(log_M, log_r.max().item()), 0.0, 1e-6)
    keep = torch.rand(2000, generator=rng).double() <= torch.sigmoid(f_hat - torch.quantile(f_hat, percentile / 100))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
    for ax, images, title in zip(axes, (x[keep][:n], x[~keep][:n]), ('Kept', 'Rejected')):
        image_grid(ax, images, title)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_kept_rejected.png', dpi=200)


def plot_two_measures(rows):
    """FID de la baseline selon les deux réseaux, chaque minimum annoté."""
    rows = list(rows.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    panels = (('fid_prof', 'Inception (trained on ImageNet)', (40, 60, 100, 150, 250)), ('fid', 'Small CNN (trained on MNIST)', (50, 100, 200, 500, 1000, 2000)))
    for ax, (key, title, ticks) in zip(axes, panels):
        best = min(rows, key=lambda r: r[key])
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=INK, linewidth=2)
        ax.scatter([best['epoch']], [best[key]], color=ORANGE, s=70, zorder=3)
        ax.annotate(f"best: epoch {best['epoch']}", (best['epoch'], best[key]), xytext=(0, -13), textcoords='offset points',
                    ha='right' if best['epoch'] > 150 else 'center', va='top', color=INK)
        ax.set(yscale='log', xlabel='Epoch', ylabel='FID', xlim=(0, 205), ylim=(best[key] / 1.9, max(r[key] for r in rows) * 1.2))
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_title(title, loc='left', fontsize=16, color=INK)
        ax.grid(axis='y', color=GRID)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_two_measures.png', dpi=200)


def bar_panels(configs, panels, path):
    """Trois panneaux de barres horizontales partageant les mêmes lignes, valeurs écrites, axes partant de 0.

    configs : (étiquette, couleur, {colonne : valeur}) ; panels : (colonne, titre, format).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.9), sharey=True)
    for ax, (key, title, fmt) in zip(axes, panels):
        values = [c[2][key] for c in configs]
        ax.barh([c[0] for c in configs], values, color=[c[1] for c in configs], height=0.68)
        for y, value in enumerate(values):
            ax.annotate(fmt.format(value), (value, y), xytext=(6, 0), textcoords='offset points', va='center', color=INK)
        ax.set(xlim=(0, max(values) * 1.35))
        ax.set_title(title, loc='left', fontsize=14, color=INK)
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_data_results():
    """Résultats de Nicodème à 60 000 mises à jour (figures/data/summary.csv)."""
    with open('figures/data/summary.csv', newline='') as f:
        rows = {r['variant']: r for r in csv.DictReader(f)}
    variants = (('Reference', 'data_ref', MUTED), ('Noise 0.1', 'noise_01', AQUA), ('Noise 0.3', 'noise_03', AQUA),
                ('Selection 95%', 'sel_95', BLUE), ('Selection 90%', 'sel_90', BLUE), ('Selection 75%', 'sel_75', BLUE),
                ('Selection 50%', 'sel_50', BLUE))
    configs = [(label, color, {k: float(rows[v][k]) for k in ('fid_60k', 'precision_60k', 'recall_60k')}) for label, v, color in variants]
    bar_panels(configs, (('fid_60k', 'CNN FID', '{:.0f}'), ('precision_60k', 'CNN precision', '{:.2f}'), ('recall_60k', 'CNN recall', '{:.2f}')),
               f'{OUT}/cmp_data_results.png')


def plot_training_results():
    """Campagne finale de Noé (results.csv) : moyenne sur 3 seeds, sans la seed effondrée de la baseline (FID 2770)."""
    with open('results.csv', newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    variants = (('BCE loss (2 of 3 seeds)', 'ema100_baseline_s', MUTED), ('RpGAN + R1, γ = 1', 'ema100_rpgan_r1_s', ORANGE),
                ('R3GAN, γ = 1', 'ema100_r3gan_s', AQUA), ('RpGAN + R1, γ = 10', 'ema100_rpgan_r1_g10_s', MUTED))
    configs = []
    for label, prefix, color in variants:
        group = [r for r in rows if prefix in r['run'] and float(r['fid']) < 500]
        configs.append((label, color, {k: np.mean([float(r[k]) for r in group]) for k in ('fid', 'precision', 'recall')}))
        print(label, len(group), {k: round(v, 3) for k, v in configs[-1][2].items()})
    bar_panels(configs, (('fid', 'CNN FID', '{:.1f}'), ('precision', 'CNN precision', '{:.2f}'), ('recall', 'CNN recall', '{:.2f}')),
               f'{OUT}/cmp_training_results.png')


def load_sampling_rows(path):
    """Lignes de l'époque 200 contre la validation, dans un fichier de résultats de DRS."""
    with open(path, newline='') as f:
        return [r for r in csv.DictReader(f) if r['run'] == RUN and r['reference'] == 'val' and r['weights'] == 'best']


def select_sampling(rows, percentile):
    """Lignes sans tri (percentile None) ou du DRS à ce percentile."""
    if percentile is None:
        return [r for r in rows if r['method'] == 'plain']
    return [r for r in rows if r['method'] == 'drs_azadi' and json.loads(r['params'])['percentile'] == percentile]


def plot_sampling_results():
    """DRS sur l'époque 200, sans tri puis à chaque percentile : FID Inception (results_prof.csv), mesures du petit CNN (results.csv)."""
    cnn, inception = load_sampling_rows('results/results.csv'), load_sampling_rows('results/results_prof.csv')
    configs = []
    for percentile in (None, 50, 80, 95, 99):
        group = select_sampling(cnn, percentile)
        values = {k: np.mean([float(r[k]) for r in group]) for k in ('fid', 'precision', 'recall', 'acceptance_rate')}
        values['fid_inception'] = np.mean([float(r['fid']) for r in select_sampling(inception, percentile)])
        label = 'No rejection' if percentile is None else f"Percentile {percentile}: {100 * values['acceptance_rate']:.0f}% kept"
        color = MUTED if percentile is None else BLUE if percentile == 80 else LIGHT_BLUE
        configs.append((label, color, values))
        print(label, {k: round(v, 3) for k, v in values.items()})
    bar_panels(configs, (('fid_inception', 'Inception FID', '{:.1f}'), ('fid', 'CNN FID', '{:.0f}'),
                         ('precision', 'CNN precision', '{:.3f}'), ('recall', 'CNN recall', '{:.3f}')),
               f'{OUT}/cmp_sampling_results.png')


def plot_combined_results():
    """Protocole commun (results/comparison_local.csv, campagne de Noé) : moyennes sur 3 seeds, contre MNIST test."""
    with open(RESULTS, newline='') as f:
        rows = list(csv.DictReader(f))
    variants = (('Baseline', 'Baseline commune', MUTED), ('Baseline + DRS', 'Baseline commune + DRS', BLUE),
                ('RpGAN + R1 + EMA', 'RpGAN + R1 + EMA', ORANGE), ('+ DRS', 'RpGAN + R1 + EMA + DRS', BLUE),
                ('+ noise 0.3', 'RpGAN + R1 + EMA + bruit 0,3', AQUA), ('+ noise 0.3 + DRS', 'RpGAN + R1 + EMA + bruit 0,3 + DRS', BLUE))
    keys = ('fid', 'fid_cnn', 'precision_cnn', 'recall_cnn')
    configs = []
    for label, name, color in variants:
        group = [r for r in rows if r['name'] == name]
        configs.append((label, color, {k: np.mean([float(r[k]) for r in group]) for k in keys}))
        print(label, len(group), {k: round(v, 3) for k, v in configs[-1][2].items()})
    bar_panels(configs, (('fid', 'Inception FID', '{:.1f}'), ('fid_cnn', 'CNN FID', '{:.1f}'), ('precision_cnn', 'CNN precision', '{:.2f}'),
                         ('recall_cnn', 'CNN recall', '{:.2f}')), f'{OUT}/cmp_combined_results.png')


@torch.no_grad()
def plot_combined_samples(weights):
    """Protocole commun, seed 0 : mêmes z, sauf pour le DRS qui garde une partie des propositions."""
    z = fixed_noise()
    G = load_model(Generator(784), os.path.join(weights, 'cmploc_baseline_G.pth'), 'cpu').eval()
    D = load_model(Discriminator(784), os.path.join(weights, 'cmploc_baseline_D.pth'), 'cpu').eval()
    grids = [('Baseline', G(z)), ('Baseline + DRS', sample(G, D, len(z), {'method': 'drs_azadi', 'percentile': 80}, seed=0, batch_size=1000)[0])]
    for title, name in (('RpGAN + R1 + EMA', 'cmploc_rpgan_r1_ema_G.pth'), ('+ noise 0.3', 'cmploc_rpgan_r1_ema_noise_03_G.pth')):
        grids.append((title, load_model(Generator(784), os.path.join(weights, name), 'cpu').eval()(z)))
    fig, axes = plt.subplots(1, len(grids), figsize=(12, 3.5))
    for ax, (title, images) in zip(axes, grids):
        image_grid(ax, images, title)
    fig.tight_layout()
    fig.savefig(f'{OUT}/cmp_combined_samples.png', dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the comparison slides figures.')
    parser.add_argument('--weights', required=True, help='Dossier contenant les générateurs (voir en haut du fichier).')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    evaluations = load_evaluations()
    plot_epochs(evaluations, args.weights)
    plot_digits(evaluations)
    plot_selection()
    plot_data_samples(args.weights)
    plot_loss_effect(args.weights)
    plot_kept_rejected()
    plot_two_measures(evaluations)
    plot_data_results()
    plot_training_results()
    plot_sampling_results()
    plot_combined_results()
    plot_combined_samples(args.weights)
