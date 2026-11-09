"""Tableau et tests de la partie Inference (DRS / OBRS), depuis drs_recalib_test.csv (Juliet, 3 seeds x 2 tirages, test).

Usage, depuis rapport/report/ :
    python figscripts/inference.py
Sortie standard : les lignes LaTeX du tableau tab:inference-results (moyenne sur les tirages, puis moyenne +- écart-type
sur les seeds), les temps médians, puis les tests t appariés par seed cités dans le texte (paired_test).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'enquete', 'results', 'juliet', 'drs_recalib_test.csv')
MNIST_TEST = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]) / 10000
G_NAMES = {'baseline': 'Baseline (BCE)', 'rpgan_r1_ema': 'RpGAN + R1 + EMA'}
VARIANTS = {'plain': 'none', 'drs_p80': 'DRS', 'obrs_K3': 'OBRS', 'drs_p80_recal': 'DRS, recal.',
            'obrs_K3_recal': 'OBRS, recal.'}


def digit_tv(classes):
    """Distance en variation totale entre le mélange de chiffres '1220-2717-...' et celui de MNIST test."""
    counts = np.array(classes.split('-'), float)
    return 0.5 * np.abs(counts / counts.sum() - MNIST_TEST).sum()


def per_seed():
    """Une ligne par (G, variante, seed) : métriques moyennées sur les 2 tirages."""
    d = pd.read_csv(CSV)
    d['g'], d['seed'] = d.model.str.rsplit('_', n=1).str[0], d.model.str.rsplit('_', n=1).str[1]
    d['tv'] = d.classes.map(digit_tv)
    cols = ['fid', 'precision', 'recall', 'fid_cnn', 'acceptance_rate', 'seconds', 'auc', 'tv']
    return d.groupby(['g', 'variant', 'seed'])[cols].mean().reset_index()


def table_rows(s):
    """Lignes LaTeX : rejet, AUC, acceptation, FID_Inc (moy +- é.-t.), P/R, FID_CNN, TV."""
    m, sd = s.groupby(['g', 'variant']).mean(numeric_only=True), s.groupby(['g', 'variant']).std(numeric_only=True)
    for g, name in G_NAMES.items():
        print(rf'\multicolumn{{7}}{{l}}{{\textit{{{name}}}}}\\')
        for v, label in VARIANTS.items():
            r, e = m.loc[(g, v)], sd.loc[(g, v)]
            auc = '--' if v == 'plain' else f'{r.auc:.2f}'
            cnn = f'{r.fid_cnn:.0f}' if g == 'baseline' else f'{r.fid_cnn:.1f}'
            print(rf'{label} & {auc} & {100 * r.acceptance_rate:.0f} & {r.fid:.1f}$\pm${e.fid:.1f} & '
                  rf'{r.precision:.2f}/{r.recall:.2f} & {cnn} & {r.tv:.3f} \\')
    print('seconds (median per variant):', s.groupby('variant').seconds.median().round(3).to_dict())


def paired_test(s, g, a, b, col='fid'):
    """Différences par seed (b - a) de la métrique col pour le G g, leur moyenne et la p-value du test t apparié."""
    va, vb = (s[(s.g == g) & (s.variant == v)].sort_values('seed')[col].values for v in (a, b))
    print(f'{g} {col}: {a} -> {b}', (vb - va).round(3), f'mean {(vb - va).mean():+.3f}', f'p={ttest_rel(vb, va).pvalue:.3f}')


TESTS = [('baseline', 'plain', 'drs_p80', 'fid'), ('baseline', 'drs_p80', 'obrs_K3', 'fid'),
         ('baseline', 'drs_p80', 'drs_p80_recal', 'fid'), ('rpgan_r1_ema', 'plain', 'drs_p80', 'fid'),
         ('rpgan_r1_ema', 'plain', 'drs_p80', 'fid_cnn'), ('rpgan_r1_ema', 'plain', 'drs_p80', 'tv'),
         ('rpgan_r1_ema', 'plain', 'drs_p80_recal', 'fid'), ('rpgan_r1_ema', 'plain', 'obrs_K3_recal', 'fid'),
         ('rpgan_r1_ema', 'plain', 'obrs_K3_recal', 'precision')]


if __name__ == '__main__':
    seeds = per_seed()
    table_rows(seeds)
    for test in TESTS:
        paired_test(seeds, *test)
