"""Corrélation de rang (Spearman) entre FID Inception et FID du petit CNN, sur toutes les mesures appariées du dépôt.

Sources lues (lecture seule, worktree wt-comparaison) :
    runs/baseline_200ep_prof/metrics.jsonl : une mesure par époque paire, mêmes images, champs fid (CNN) et fid_prof ;
    results/comparison_local.csv           : protocole commun contre test, colonnes fid (Inception) et fid_cnn ;
    results/results.csv + results_prof.csv : appariées sur (run, weights, reference, method, params, seed).
Lancer depuis le worktree :  python <scratchpad>/scripts/fid/rank_correlation.py <scratchpad>
Sortie : <scratchpad>/results/fid/rank_correlation.csv
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

OUT = os.path.join(sys.argv[1], 'results', 'fid', 'rank_correlation.csv')


def jsonl_pairs(path, min_epoch=0):
    """(fid Inception, fid CNN) de chaque époque mesurée avec les deux réseaux, à partir de min_epoch."""
    rows = [json.loads(line) for line in open(path)]
    return [(r['fid_prof'], r['fid']) for r in rows if 'fid_prof' in r and r['epoch'] >= min_epoch]


def comparison_pairs(path, drop_drs=False, only=''):
    """(fid Inception, fid CNN) de results/comparison_local.csv ; `only` garde les lignes dont le nom le contient."""
    rows = list(csv.DictReader(open(path)))
    return [(float(r['fid']), float(r['fid_cnn'])) for r in rows
            if not (drop_drs and 'DRS' in r['name']) and only in r['name']]


def class_gap(counts):
    """Part des images qui devraient changer de chiffre pour atteindre 10 % par chiffre (comme baseline.md)."""
    c = np.array([int(v) for v in counts.split('-')], dtype=float)
    return 0.5 * np.abs(c / c.sum() - 0.1).sum()


def gap_pairs(path, column):
    """(écart des chiffres, FID `column`) de results/comparison_local.csv."""
    return [(class_gap(r['classes']), float(r[column])) for r in csv.DictReader(open(path))]


def results_pairs(path_cnn, path_prof):
    """Lignes de results.csv et results_prof.csv qui décrivent la même évaluation (mêmes images)."""
    key = lambda r: (r['run'], r['weights'], r['reference'], r['method'], r['params'], r['seed'])
    prof = {key(r): float(r['fid']) for r in csv.DictReader(open(path_prof))}
    return [(prof[key(r)], float(r['fid'])) for r in csv.DictReader(open(path_cnn)) if key(r) in prof]


def describe(name, pairs):
    """Ligne de résultat : n, rho de Spearman, p-valeur, plages des deux colonnes (x = 1re, y = 2e)."""
    a = np.array(pairs)
    rho, p = spearmanr(a[:, 0], a[:, 1])
    return dict(source=name, n=len(a), spearman=round(rho, 3), p_value=f'{p:.2g}',
                inception_min=round(a[:, 0].min(), 1), inception_max=round(a[:, 0].max(), 1),
                cnn_min=round(a[:, 1].min(), 1), cnn_max=round(a[:, 1].max(), 1))


if __name__ == '__main__':
    sets = {
        'baseline_200ep_prof, toutes epoques': jsonl_pairs('runs/baseline_200ep_prof/metrics.jsonl'),
        'baseline_200ep_prof, epoques >= 100': jsonl_pairs('runs/baseline_200ep_prof/metrics.jsonl', 100),
        'comparison_local (18 lignes)': comparison_pairs('results/comparison_local.csv'),
        'comparison_local sans DRS (9)': comparison_pairs('results/comparison_local.csv', drop_drs=True),
        'comparison_local RpGAN seulement (12)': comparison_pairs('results/comparison_local.csv', only='RpGAN'),
        'results.csv x results_prof.csv': results_pairs('results/results.csv', 'results/results_prof.csv'),
        'ecart des chiffres vs FID CNN (comparison_local)': gap_pairs('results/comparison_local.csv', 'fid_cnn'),
        'ecart des chiffres vs FID Inception (comparison_local)': gap_pairs('results/comparison_local.csv', 'fid'),
    }
    rows = [describe(name, pairs) for name, pairs in sets.items() if len(pairs) > 2]
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
