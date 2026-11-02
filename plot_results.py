"""Figures du rapport à partir de results/results.csv (moyenne ± écart-type sur les seeds).

    figures/pr_plane.pdf      plan (rappel, précision) : une courbe par méthode, points annotés par K ou psi
    figures/pr_plane_zoom.pdf idem, limité à plain / OBRS / DRS (la truncation écrase l'échelle)
    figures/fid_budget.pdf    FID en fonction du budget K = 1 / taux d'acceptation (OBRS vs DRS, plain à K = 1)
    figures/classes.pdf       part de chaque chiffre 0-9 : réel (1/10), plain, OBRS / DRS à --K, truncation à --psi

Utilisation :
    python plot_results.py --run checkpoints --reference val            # classes.pdf : K = 10, psi = 1.5
    python plot_results.py --run runs/baseline_v1 --weights best --reference test
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate import load_results, summarize

COLORS = {'plain': 'black', 'obrs': 'tab:red', 'drs': 'tab:blue', 'truncation': 'tab:green'}
MAIN_PARAM = {'obrs': 'K', 'drs': 'K', 'truncation': 'psi'}


def select(rows, run, weights, reference):
    """Lignes d'un run, d'un jeu de poids et d'une référence donnés."""
    return [r for r in rows if r['run'] == run and r['weights'] == weights and r['reference'] == reference]


def by_method(summary):
    """{méthode: [(valeur du paramètre principal, agrégats), ...] triés par valeur}."""
    out = {}
    for (_, _, _, method, params), agg in summary.items():
        value = json.loads(params).get(MAIN_PARAM.get(method), 0)
        out.setdefault(method, []).append((value, agg))
    return {m: sorted(points, key=lambda p: p[0]) for m, points in out.items()}


def plot_pr_plane(methods, path, connect=True):
    """Plan (rappel, précision) : un point par réglage (moyenne ± écart-type), reliés si connect."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (method, points) in enumerate(methods.items()):
        r = [a['recall_mean'] for _, a in points]
        p = [a['precision_mean'] for _, a in points]
        if method == 'plain':
            ax.scatter(r, p, marker='x', s=80, color='black', label='plain (G seul)', zorder=3)
            continue
        ax.errorbar(r, p, xerr=[a['recall_std'] for _, a in points], yerr=[a['precision_std'] for _, a in points],
                    marker='o', ls='-' if connect else 'none', color=COLORS[method], label=method.upper(),
                    capsize=2)
        for value, a in points:
            ax.annotate(f"{MAIN_PARAM[method]}={value:g}", (a['recall_mean'], a['precision_mean']), fontsize=7,
                        xytext=(4, 6 - 12 * (i % 2)), textcoords='offset points', color=COLORS[method])
    ax.set(xlabel='Rappel (diversité)', ylabel='Précision (qualité)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def plot_fid_budget(methods, path):
    """FID moyen (± écart-type) en fonction du budget K = 1 / acceptation, pour les méthodes de rejet."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for method in ('obrs', 'drs'):
        points = methods.get(method, [])
        k = [1 / a['acceptance_rate_mean'] for _, a in points]
        ax.errorbar(k, [a['fid_mean'] for _, a in points], yerr=[a['fid_std'] for _, a in points],
                    marker='o', color=COLORS[method], label=method.upper(), capsize=2)
    if 'plain' in methods:
        ax.axhline(methods['plain'][0][1]['fid_mean'], color='black', ls='--', label='plain')
    budgets = sorted({value for method in ('obrs', 'drs') for value, _ in methods.get(method, [])})
    ax.set(xscale='log', xlabel='Budget K = propositions par image gardée', ylabel='FID (plus bas = mieux)')
    ax.set_xticks(budgets, [f'{k:g}' for k in budgets])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def class_shares(rows, method, value):
    """Part moyenne (%) de chaque chiffre, toutes seeds, pour une méthode et la valeur de son paramètre principal."""
    counts = np.array([[int(c) for c in r['classes'].split('-')] for r in rows if r['method'] == method
                       and json.loads(r['params']).get(MAIN_PARAM.get(method)) == value])
    return 100 * counts.sum(0) / counts.sum()


def plot_classes(rows, settings, path):
    """Histogramme des chiffres pour les réglages choisis, ex. {'plain': None, 'obrs': 10, 'truncation': 1.5}."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(settings)
    for i, (method, value) in enumerate(settings.items()):
        label = method if value is None else f'{method} {MAIN_PARAM[method]}={value:g}'
        ax.bar(np.arange(10) + i * width, class_shares(rows, method, value), width, label=label,
               color=COLORS[method])
    ax.axhline(10, color='gray', ls=':', label='réel ≈ 10 %')
    ax.set(xticks=np.arange(10) + 0.4 - width / 2, xticklabels=range(10), xlabel='Chiffre prédit',
           ylabel='Part des images (%)')
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot report figures from results/results.csv.')
    parser.add_argument('--run', required=True, help='Run à tracer (tel qu\'écrit dans la colonne run).')
    parser.add_argument('--weights', choices=['last', 'best'], default='last')
    parser.add_argument('--reference', choices=['val', 'test'], default='val')
    parser.add_argument('--out', default='figures', help='Dossier des figures.')
    parser.add_argument('--metric', choices=['cnn', 'prof'], default='cnn', help='Tableau de résultats à tracer.')
    parser.add_argument('--K', type=float, default=10, help='Budget OBRS / DRS montré dans classes.pdf.')
    parser.add_argument('--psi', type=float, default=1.5, help='Truncation montrée dans classes.pdf.')
    args = parser.parse_args()

    rows = select(load_results(args.metric), args.run, args.weights, args.reference)
    summary = summarize(rows)
    methods = by_method(summary)
    os.makedirs(args.out, exist_ok=True)
    plot_pr_plane(methods, os.path.join(args.out, 'pr_plane.pdf'))
    rejection_only = {m: p for m, p in methods.items() if m in ('plain', 'obrs', 'drs')}
    plot_pr_plane(rejection_only, os.path.join(args.out, 'pr_plane_zoom.pdf'), connect=False)
    plot_fid_budget(methods, os.path.join(args.out, 'fid_budget.pdf'))
    settings = {'plain': None, 'obrs': args.K, 'drs': args.K, 'truncation': args.psi}
    plot_classes(rows, {m: v for m, v in settings.items() if m in methods}, os.path.join(args.out, 'classes.pdf'))
    print(f'Figures écrites dans {args.out}/ ({len(rows)} lignes, méthodes : {sorted(methods)})')
