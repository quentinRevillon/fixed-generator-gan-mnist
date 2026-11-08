"""Recalcule les chiffres des slides (comparaison.tex) depuis les CSV/JSONL du worktree wt-comparaison.

Utilisation : python audit.py <chemin du worktree>
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

WT = sys.argv[1]


def read_csv(path):
    """Lignes d'un CSV en dictionnaires."""
    with open(f'{WT}/{path}', newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def ms(values):
    """Moyenne, écart-type (ddof=1), n."""
    v = np.array(values, float)
    return round(v.mean(), 2), round(v.std(ddof=1), 2) if len(v) > 1 else 0.0, len(v)


print('=== Baseline 200 ep (metrics.jsonl, validation) ===')
rows = {}
with open(f'{WT}/runs/baseline_200ep_prof/metrics.jsonl') as f:
    for r in map(json.loads, f):
        if 'fid_prof' in r:
            rows[r['epoch']] = r
for e in (50, 106, 120, 200):
    if e in rows:
        r = rows[e]
        print(e, 'incep', round(r['fid_prof'], 1), 'cnn', round(r['fid'], 1), 'P/R cnn', round(r['precision'], 3), round(r['recall'], 3),
              'P/R vgg', round(r['precision_prof'], 3), round(r['recall_prof'], 3))
be_cnn = min(rows, key=lambda e: rows[e]['fid'])
be_inc = min(rows, key=lambda e: rows[e]['fid_prof'])
print('best cnn epoch', be_cnn, round(rows[be_cnn]['fid'], 1), '| best inception epoch', be_inc, round(rows[be_inc]['fid_prof'], 1))
print('evaluated epochs', len(rows), 'from', min(rows), 'to', max(rows))
for e in (be_cnn, be_inc):
    c = np.array(rows[e]['classes'])
    print('classes epoch', e, np.round(100 * c / c.sum(), 1))

print('\n=== Common protocol (results/comparison_local.csv, test) ===')
cmp = read_csv('results/comparison_local.csv')
groups = defaultdict(list)
for r in cmp:
    groups[r['name']].append(r)
for name, g in groups.items():
    out = {k: ms([float(r[k]) for r in g]) for k in ('fid', 'fid_cnn', 'precision', 'recall', 'precision_cnn', 'recall_cnn', 'acceptance_rate')}
    print(name, out)

print('\nWelch t-tests and paired (same training seed) differences:')
def vals(name, k):
    return np.array([float(r[k]) for r in sorted(groups[name], key=lambda r: r['run'])])
pairs = [('RpGAN + R1 + EMA', 'RpGAN + R1 + EMA + bruit 0,3', False), ('RpGAN + R1 + EMA', 'RpGAN + R1 + EMA + DRS', True),
         ('Baseline commune', 'Baseline commune + DRS', True), ('RpGAN + R1 + EMA + bruit 0,3', 'RpGAN + R1 + EMA + bruit 0,3 + DRS', True),
         ('Baseline commune', 'RpGAN + R1 + EMA', False)]
for a, b, paired in pairs:
    for k in ('fid', 'fid_cnn', 'precision', 'recall'):
        x, y = vals(a, k), vals(b, k)
        t = stats.ttest_rel(x, y) if paired else stats.ttest_ind(x, y, equal_var=False)
        d = y - x
        print(f'{a} -> {b} [{k}] diff mean {d.mean():+.2f}' + (f' per-seed {np.round(d, 2)}' if paired else '') +
              f' p={t.pvalue:.3f} ({"paired" if paired else "welch"})')

print('\n=== Training campaign (results.csv, 60k, 100 ep, EMA, test) ===')
tr = read_csv('results.csv')
for prefix in ('ema100_baseline_s', 'ema100_rpgan_r1_s', 'ema100_r3gan_s', 'ema100_rpgan_r1_g10_s'):
    g = [r for r in tr if prefix in r['run']]
    ok = [r for r in g if float(r['fid']) < 500]
    print(prefix, 'all', ms([float(r['fid']) for r in g]), 'stable', {k: ms([float(r[k]) for r in ok]) for k in ('fid', 'precision', 'recall')})
x = [float(r['fid']) for r in tr if 'ema100_rpgan_r1_s' in r['run']]
print('rpgan_r1 g1 min/max', min(x), max(x), 'half-range', round((max(x) - min(x)) / 2, 2))
for prefix in ('ema100_rpgan_r1_s', 'ema100_r3gan_s', 'ema100_rpgan_r1_g10_s'):
    pass
a = [float(r['fid']) for r in tr if 'ema100_rpgan_r1_s' in r['run']]
b = [float(r['fid']) for r in tr if 'ema100_rpgan_r1_g10_s' in r['run']]
c = [float(r['fid']) for r in tr if 'ema100_r3gan_s' in r['run']]
print('g1 vs g10 welch p', round(stats.ttest_ind(a, b, equal_var=False).pvalue, 4), '| rpgan vs r3gan p', round(stats.ttest_ind(a, c, equal_var=False).pvalue, 3))
bce = [r for r in tr if 'ema100_baseline_s' in r['run'] and float(r['fid']) < 500]
for name, g in (('BCE', bce), ('RpGAN', [r for r in tr if 'ema100_rpgan_r1_s' in r['run']])):
    c = np.array([[int(v) for v in r['classes'].split('-')] for r in g], float)
    print('shares', name, np.round(100 * (c / c.sum(1, keepdims=True)).mean(0), 1))

print('\n=== DRS on baseline epoch 200 (results/results*.csv, val) ===')
def sampling_rows(path):
    return [r for r in read_csv(path) if r['run'] == 'runs/baseline_200ep_prof' and r['reference'] == 'val' and r['weights'] == 'best']
cnn, inc = sampling_rows('results/results.csv'), sampling_rows('results/results_prof.csv')
def select(rows, p):
    if p is None:
        return [r for r in rows if r['method'] == 'plain']
    return [r for r in rows if r['method'] == 'drs_azadi' and json.loads(r['params']).get('percentile') == p]
for p in (None, 50, 80, 95, 99):
    g, h = select(cnn, p), select(inc, p)
    print(p, 'cnn n', len(g), {k: ms([float(r[k]) for r in g]) for k in ('fid', 'precision', 'recall', 'acceptance_rate')},
          'incep n', len(h), ms([float(r['fid']) for r in h]), 'seeds', sorted({r['seed'] for r in h}), 'params', sorted({r['params'] for r in g}))

print('\n=== Data (figures/data/summary.csv and runs.csv) ===')
for r in read_csv('figures/data/summary.csv'):
    print(r['variant'], r['seeds'], r['collapsed'], r['fid_60k'], r['fid_60k_std'], r['precision_60k'], r['recall_60k'], r['fid_final'])
runs = read_csv('figures/data/runs.csv')
by = defaultdict(list)
for r in runs:
    by[r['variant']].append(r)
print({k: (len(v), sum(int(r['collapsed']) for r in v)) for k, v in by.items()})
ok = lambda v: [float(r['fid_60k']) for r in by[v] if r['collapsed'] == '0']
for v in ('noise_01', 'noise_03', 'sel_95'):
    print('ref vs', v, 'welch p', round(stats.ttest_ind(ok('data_ref'), ok(v), equal_var=False).pvalue, 3), ms(ok('data_ref')), ms(ok(v)))
# Fisher tests on collapse rates
print('fisher 0/3 noise03 vs 3/8 ref', stats.fisher_exact([[0, 3], [3, 5]]).pvalue)
print('fisher 0/8 noise03 vs 15/34 others', stats.fisher_exact([[0, 8], [15, 19]]).pvalue)
print('fisher 0/3 noise03 vs 15/34 others', stats.fisher_exact([[0, 3], [15, 19]]).pvalue)
print('fisher ref 3/8 vs selection 8/18', stats.fisher_exact([[3, 5], [8, 10]]).pvalue)
