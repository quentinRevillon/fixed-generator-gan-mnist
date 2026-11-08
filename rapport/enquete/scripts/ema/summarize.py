"""Tableau EMA vs perte : FID Inception (cache .npy de inception_feats.py ou comparison_local.csv), FID CNN, P/R.

Utilisation :
    DATA=<data> python summarize.py <code_dir> <cache_dir> <out_dir>
Sorties : <out_dir>/ema_per_seed.csv (une ligne par modèle et seed), <out_dir>/ema_summary.csv (moyenne ± écart-type
sur les seeds, écart-type échantillon ddof=1), <out_dir>/ema_welch.csv (tests de Welch).
FID Inception : mesuré ici si le .npy existe, sinon repris de results/comparison_local.csv (même chaîne, N = 10 000,
z de la seed 0 ; voir le contrôle de cohérence). FID CNN et P/R CNN (k = 3) : recalculés ici contre MNIST test.
P/R VGG : seulement ceux de comparison_local.csv (VGG trop lent sur CPU pour G_raw).
"""
import csv
import os
import sys

import numpy as np
import torch
from scipy import stats

torch.set_num_threads(int(os.getenv('N_THREADS', 4)))
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from evaluate import evaluate, fid, generate  # noqa: E402
from model import Generator  # noqa: E402
from utils import load_model  # noqa: E402

DEVICE = torch.device('cpu')
N = 10_000
SEEDS = (0, 1, 2)
MODELS = {  # nom -> (préfixe du run, fichier de poids)
    'Baseline (G, sans EMA)': ('cmploc_baseline', 'G.pth'),
    'RpGAN+R1 sans EMA (G_raw)': ('cmploc_rpgan_r1_ema', 'G_raw.pth'),
    'RpGAN+R1 + EMA (G)': ('cmploc_rpgan_r1_ema', 'G.pth'),
    'RpGAN+R1 + bruit 0,3 sans EMA (G_raw)': ('cmploc_rpgan_r1_ema_noise_03', 'G_raw.pth'),
    'RpGAN+R1 + bruit 0,3 + EMA (G)': ('cmploc_rpgan_r1_ema_noise_03', 'G.pth'),
}
WELCH = [('Baseline (G, sans EMA)', 'RpGAN+R1 sans EMA (G_raw)'),
         ('RpGAN+R1 sans EMA (G_raw)', 'RpGAN+R1 + EMA (G)'),
         ('Baseline (G, sans EMA)', 'RpGAN+R1 + EMA (G)'),
         ('RpGAN+R1 + bruit 0,3 sans EMA (G_raw)', 'RpGAN+R1 + bruit 0,3 + EMA (G)')]


def group_rows():
    """Lignes sans DRS de results/comparison_local.csv, indexées par chemin du G (runs/<run>/G.pth)."""
    with open('results/comparison_local.csv', newline='') as f:
        return {r['run']: r for r in csv.DictReader(f) if 'DRS' not in r['name']}


def inception_fid(cache_dir, run, weights, f_real):
    """FID Inception depuis le cache, ou None si les features de ce G n'ont pas été calculées."""
    path = os.path.join(cache_dir, f'{run}__{weights}__n{N}.npy')
    return fid(f_real, torch.from_numpy(np.load(path))) if os.path.exists(path) else None


def cnn_metrics(run, weights):
    """FID CNN, precision, recall (petit CNN, k = 3) de N images du G, contre MNIST test."""
    G = load_model(Generator(784).to(DEVICE), os.path.join('runs', run, weights), DEVICE)
    return evaluate(generate(G, N, DEVICE), DEVICE, reference='test')


def seed_row(name, run, weights, cache_dir, f_real, group):
    """Une ligne du tableau par seed : Inception mesuré et/ou du groupe, CNN recalculé, VGG du groupe."""
    g = group.get(f'runs/{run}/{weights}', {})
    measured = inception_fid(cache_dir, run, weights, f_real)
    m = cnn_metrics(run, weights)
    inc = measured if measured is not None else (float(g['fid']) if g else float('nan'))
    return {'model': name, 'run': run, 'weights': weights, 'n': N,
            'fid_inception': inc, 'fid_inception_source': 'mesure' if measured is not None else 'comparison_local',
            'fid_inception_groupe': g.get('fid', ''), 'fid_cnn': m['fid'],
            'precision_cnn': m['precision'], 'recall_cnn': m['recall'],
            'precision_vgg_groupe': g.get('precision', ''), 'recall_vgg_groupe': g.get('recall', '')}


def mean_std(values):
    """(moyenne, écart-type échantillon) en ignorant les NaN et les vides."""
    v = np.array([float(x) for x in values if x != ''], dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else float('nan'), len(v)) if len(v) else (float('nan'),) * 2 + (0,)


def write_csv(path, rows):
    """Écrit une liste de dicts en CSV."""
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    cache_dir, out_dir = sys.argv[2], sys.argv[3]
    f_real = torch.from_numpy(np.load(os.path.join(cache_dir, 'real_test.npy')))
    group = group_rows()
    rows = [seed_row(name, f'{prefix}_s{s}', weights, cache_dir, f_real, group)
            for name, (prefix, weights) in MODELS.items() for s in SEEDS]
    write_csv(os.path.join(out_dir, 'ema_per_seed.csv'), rows)
    metrics = ['fid_inception', 'fid_cnn', 'precision_cnn', 'recall_cnn', 'precision_vgg_groupe', 'recall_vgg_groupe']
    summary = []
    for name in MODELS:
        sel = [r for r in rows if r['model'] == name]
        out = {'model': name, 'sources_inception': '/'.join(r['fid_inception_source'] for r in sel)}
        for k in metrics:
            out[f'{k}_mean'], out[f'{k}_std'], out[f'{k}_n'] = mean_std([r[k] for r in sel])
        summary.append(out)
    write_csv(os.path.join(out_dir, 'ema_summary.csv'), summary)
    welch = []
    for a, b in WELCH:
        for k in ('fid_inception', 'fid_cnn'):
            va = [r[k] for r in rows if r['model'] == a and not np.isnan(r[k])]
            vb = [r[k] for r in rows if r['model'] == b and not np.isnan(r[k])]
            t, p = stats.ttest_ind(va, vb, equal_var=False) if min(len(va), len(vb)) > 1 else (np.nan, np.nan)
            welch.append({'a': a, 'b': b, 'metric': k, 'n_a': len(va), 'n_b': len(vb),
                          'mean_a': np.mean(va), 'mean_b': np.mean(vb), 't': t, 'p': p})
    write_csv(os.path.join(out_dir, 'ema_welch.csv'), welch)
    for r in rows:
        print(r['model'], r['run'], f"inc={r['fid_inception']:.2f} ({r['fid_inception_source']}, groupe={r['fid_inception_groupe']})",
              f"cnn={r['fid_cnn']:.2f} P={r['precision_cnn']:.3f} R={r['recall_cnn']:.3f}")
    for w in welch:
        print(w)
