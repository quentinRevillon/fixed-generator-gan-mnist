"""Figures de inference-drs.md dans figures/inference/, et chiffres à reporter dans le document.

Lit results/results.csv, results/results_prof.csv, les keep_training.jsonl et les poids de runs/baseline_200ep_prof.

Utilisation, depuis la racine du repo :
    python figures/make_inference_figures.py
"""
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

from evaluate import load_results, summarize, to_unit_images
from model import Discriminator, Generator
from sampling import calibration_pool, drs_f_hat, propose
from utils import load_model

OUT = 'figures/inference'
RUN_200, RUN_120 = 'runs/baseline_200ep_prof', 'runs/baseline_200ep'
PERCENTILES = (50, 80, 95, 99)
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def plain_point(summary, run):
    """Agrégats d'un run sans tri."""
    return next(a for k, a in summary.items() if k[0] == run and k[3] == 'plain' and k[2] == 'val')


def percentile_point(summary, run, weights, percentile):
    """Agrégats de drs_azadi pour un run, un jeu de poids et un percentile."""
    return next(a for k, a in summary.items() if k[0] == run and k[1] == weights and k[3] == 'drs_azadi'
                and json.loads(k[4])['percentile'] == percentile)


def all_points(summary, run):
    """Les configurations d'un run mesurées contre val : (nom, agrégats, couleur)."""
    points = [('Sans tri', plain_point(summary, run), MUTED)]
    points += [(f'DRS, percentile {p}', percentile_point(summary, run, 'best', p), BLUE) for p in PERCENTILES]
    points += [(f'KeepTraining + DRS, percentile {p}', percentile_point(summary, run, 'best_kt', p), ORANGE) for p in PERCENTILES]
    return points


def plot_all_metrics(points):
    """FID, précision et rappel de toutes les configurations, valeurs écrites, axes partant de 0."""
    names = [p[0] for p in points]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
    panels = (('fid', 'FID (plus bas = mieux)', '{:.1f}'), ('precision', 'Précision (plus haut = mieux)', '{:.3f}'),
              ('recall', 'Rappel (plus haut = mieux)', '{:.3f}'))
    for ax, (key, label, fmt) in zip(axes, panels):
        values = [p[1][f'{key}_mean'] for p in points]
        ax.barh(names, values, xerr=[p[1][f'{key}_std'] for p in points], color=[p[2] for p in points], height=0.72,
                error_kw={'ecolor': INK, 'elinewidth': 0.8})
        for y, value in enumerate(values):
            ax.annotate(fmt.format(value), (value, y), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK)
        ax.set(xlabel=label, xlim=(0, max(values) * 1.28))
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{OUT}/drs_all_metrics.png', dpi=200)


def class_shares(rows, weights, method, percentile=None):
    """Part moyenne (%) de chaque chiffre sur les seeds."""
    picked = [r for r in rows if r['weights'] == weights and r['method'] == method
              and (percentile is None or json.loads(r['params'])['percentile'] == percentile)]
    counts = np.array([[int(c) for c in r['classes'].split('-')] for r in picked]).sum(0)
    return 100 * counts / counts.sum()


def imbalance(shares):
    """Part des images (%) qui devrait changer de chiffre pour atteindre 10 % partout."""
    return float(np.abs(shares - 10).sum() / 2)


def class_bars(rows, percentile=80):
    """Les trois configurations comparées chiffre par chiffre : (nom, parts, couleur, motif)."""
    return (('Sans tri', class_shares(rows, 'best', 'plain'), MUTED, ''),
            (f'DRS, percentile {percentile}', class_shares(rows, 'best', 'drs_azadi', percentile), BLUE, ''),
            (f'KeepTraining + DRS, percentile {percentile}', class_shares(rows, 'best_kt', 'drs_azadi', percentile), ORANGE, '//'))


def plot_classes(bars):
    """Part de chaque chiffre pour les trois configurations."""
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    for i, (label, shares, color, hatch) in enumerate(bars):
        ax.bar(np.arange(10) + (i - 1) * 0.27, shares, 0.25, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.5, label=label)
    ax.axhline(10, color=MUTED, linestyle=':', linewidth=1.2)
    ax.annotate('réel : ≈ 10 %', (9.55, 10), xytext=(0, 3), textcoords='offset points', fontsize=9, color=MUTED)
    top = max(float(shares.max()) for _, shares, _, _ in bars)
    ax.set(xticks=range(10), xlabel='Chiffre prédit par le petit CNN', ylabel='% des images', xlim=(-0.7, 11), ylim=(0, top * 1.25))
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(f'{OUT}/drs_classes.png', dpi=200)


@torch.no_grad()
def draw_proposals(G, D, n=10_000, seed=0):
    """Propositions de G avec leur note d, le plus grand d observé (burn-in compris) et le générateur aléatoire."""
    rng = torch.Generator().manual_seed(seed)
    log_M = calibration_pool(G, D, rng, 10_000, n).max().item()
    x, log_r = propose(G, D, n, rng, n)
    return x, log_r, max(log_M, log_r.max().item()), rng


def keep_probability(d, log_M, gamma):
    """Probabilité de garder une image de note d."""
    return torch.sigmoid(drs_f_hat(d, log_M, gamma, 1e-6))


def plot_score_to_chance(log_r, log_M):
    """En haut, les notes de D ; en bas, la chance d'être gardée pour chaque note. Même axe horizontal, deux panneaux."""
    grid = torch.linspace(log_r.min().item(), log_M, 400)
    f_hat = drs_f_hat(log_r, log_M, 0.0, 1e-6)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 5), sharex=True, height_ratios=(1, 1.5))
    top.hist(log_r.numpy(), bins=70, color=MUTED, edgecolor=SURFACE, linewidth=0.3)
    top.set(ylabel='Nombre d\'images')
    top.set_title('Les notes données par D à 10 000 images de G', loc='left', fontsize=11, color=INK)
    bottom.plot(grid, torch.exp(grid.double() - log_M), color=MUTED, linestyle='--', linewidth=2, label='Formule brute, sans décalage')
    for p, color in ((50, AQUA), (80, BLUE), (95, ORANGE)):
        gamma = torch.quantile(f_hat, p / 100).item()
        bottom.plot(grid, keep_probability(grid, log_M, gamma), color=color, linewidth=2, label=f'Percentile {p}')
    bottom.set(xlabel='Note d (plus haut = plus crédible)', ylabel='Chance d\'être gardée', ylim=(0, 1.02))
    bottom.set_title('La chance d\'être gardée selon la note', loc='left', fontsize=11, color=INK)
    bottom.grid(axis='y', color=GRID)
    bottom.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(f'{OUT}/drs_score_to_chance.png', dpi=200)


def plot_kept_rejected(x, log_r, log_M, rng, percentile=80, n_show=64):
    """Mêmes propositions, séparées par la décision de DRS au percentile donné (avec le D d'origine)."""
    f_hat = drs_f_hat(log_r, log_M, 0.0, 1e-6)
    keep = torch.rand(len(x), generator=rng).double() <= torch.sigmoid(f_hat - torch.quantile(f_hat, percentile / 100))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax, images, title in zip(axes, (x[keep], x[~keep]), ('Images gardées', 'Images rejetées')):
        grid = torchvision.utils.make_grid(to_unit_images(images[:n_show]), nrow=8, padding=2, pad_value=1)
        ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
        ax.set_title(title, fontsize=11, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/drs_kept_rejected.png', dpi=200)


def plot_earlier_methods(summary):
    """OBRS et DRS réglé par K selon le taux d'acceptation, troncature selon psi (petit CNN, checkpoints/G.pth)."""
    plain = next(a for k, a in summary.items() if k[3] == 'plain')
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for method, label, color, marker in (('obrs', 'OBRS', BLUE, 'o'), ('drs', 'DRS réglé par K', ORANGE, 's')):
        pts = sorted((a2 for k, a2 in summary.items() if k[3] == method), key=lambda p: -p['acceptance_rate_mean'])
        a.errorbar([100 * p['acceptance_rate_mean'] for p in pts], [p['fid_mean'] for p in pts], yerr=[p['fid_std'] for p in pts],
                   color=color, marker=marker, markersize=5, linewidth=2, capsize=2, label=label)
    a.axhline(plain['fid_mean'], color=MUTED, linestyle='--', linewidth=1, label='Sans tri')
    a.set(xlabel='Taux d\'acceptation (%)', ylabel='FID, petit CNN', xlim=(72, -2), ylim=(0, 110))
    a.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    pts = sorted(((json.loads(k[4])['psi'], a2) for k, a2 in summary.items() if k[3] == 'truncation'), key=lambda p: p[0])
    b.plot([p for p, _ in pts], [a2['fid_mean'] for _, a2 in pts], color=AQUA, marker='o', markersize=5, linewidth=2)
    b.axhline(plain['fid_mean'], color=MUTED, linestyle='--', linewidth=1)
    b.annotate('Sans troncature', (0.5, plain['fid_mean']), xytext=(0, 4), textcoords='offset points', fontsize=9, color=MUTED)
    b.set(xlabel='Seuil ψ (plus bas = plus de troncature)', ylabel='FID, petit CNN (échelle log)', yscale='log')
    b.yaxis.set_major_formatter(ScalarFormatter())
    for ax in (a, b):
        ax.grid(axis='y', color=GRID)
    fig.tight_layout()
    fig.savefig(f'{OUT}/earlier_methods.png', dpi=200)


def format_point(name, a):
    """Une configuration sur une ligne."""
    return (f"{name}: acc {100 * a['acceptance_rate_mean']:.1f} FID {a['fid_mean']:.1f}±{a['fid_std']:.1f} "
            f"P {a['precision_mean']:.3f}±{a['precision_std']:.3f} R {a['recall_mean']:.3f}±{a['recall_std']:.3f} t {a['seconds_mean']:.2f}")


def print_keep_training(run):
    """Résumé de KeepTraining pour un run."""
    with open(f'{run}/keep_training.jsonl') as f:
        h = [json.loads(line) for line in f]
    best = min(h, key=lambda r: r['val_bce'])
    print(run, 'keep_training depart', round(h[0]['val_bce'], 3), 'meilleur', round(best['val_bce'], 3), 'pas', best['step'], 'arret', h[-1]['step'])


def print_chances(log_r, log_M):
    """Exemple chiffré : notes typiques, chance brute et chance selon le percentile."""
    f_hat = drs_f_hat(log_r, log_M, 0.0, 1e-6)
    examples = torch.quantile(log_r.double(), torch.tensor([0.1, 0.5, 0.8, 0.95], dtype=torch.double))
    print('d_M', round(log_M, 2), 'acceptation brute moyenne %', round(100 * torch.exp(log_r.double() - log_M).mean().item(), 2))
    print('notes', np.round(examples.numpy(), 2).tolist(), 'brut %', np.round(100 * torch.exp(examples - log_M).numpy(), 2).tolist())
    for p in (50, 80, 95):
        gamma = torch.quantile(f_hat, p / 100).item()
        print(f'p{p} gamma {gamma:.2f} chance % {np.round(100 * keep_probability(examples, log_M, gamma).numpy(), 1).tolist()}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows_prof, rows_cnn = load_results('prof'), load_results('cnn')
    prof, cnn = summarize(rows_prof), summarize(rows_cnn)

    points = all_points(prof, RUN_200)
    plot_all_metrics(points)
    bars = class_bars([r for r in rows_prof if r['run'] == RUN_200 and r['reference'] == 'val'])
    plot_classes(bars)
    G = load_model(Generator(784), f'{RUN_200}/G_best.pth', 'cpu').eval()
    D = load_model(Discriminator(784), f'{RUN_200}/D_best.pth', 'cpu').eval()
    x, log_r, log_M, rng = draw_proposals(G, D)
    plot_score_to_chance(log_r, log_M)
    plot_kept_rejected(x, log_r, log_M, rng)
    plot_earlier_methods({k: a for k, a in cnn.items() if k[0] == 'checkpoints' and k[2] == 'val'})

    print('== Epoque 200, Inception et VGG')
    for name, a, _ in points:
        print(format_point(name, a))
    print('== Epoque 200, petit CNN')
    print(format_point('Sans tri', plain_point(cnn, RUN_200)))
    for p in PERCENTILES:
        print(format_point(f'DRS p{p}', percentile_point(cnn, RUN_200, 'best', p)))
    print('== Epoque 120, Inception et VGG')
    for name, a, _ in all_points(prof, RUN_120):
        print(format_point(name, a))
    print('== checkpoints/G.pth, Inception et VGG, val')
    print(format_point('G.pth', plain_point(prof, 'checkpoints')))
    print('== Classes, epoque 200')
    for label, shares, _, _ in bars:
        print(label, np.round(shares, 1).tolist(), 'ecart', round(imbalance(shares), 1))
    print('== Divers')
    print_keep_training(RUN_200)
    print_keep_training(RUN_120)
    print_chances(log_r, log_M)
