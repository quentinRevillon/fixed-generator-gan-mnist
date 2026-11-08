"""Audit du FID : ce que capte le FID Inception et le FID du petit CNN, à partir des features en cache.

Pré-requis : extract_features.py (features en cache). Lancer depuis le worktree :
    python <scratchpad>/scripts/fid/analyze.py <scratchpad>
Sortie : <scratchpad>/results/fid/audit.csv (une ligne par condition et par mesure), variance_share.csv, relu.csv.

Trois réglages de mesure, comparés sur des images identiques :
    inception  : Inception, référence = 2 000 images de test, ensemble mesuré de N = 500 images (CPU lent) ;
    cnn_500    : petit CNN, mêmes images que « inception » (comparaison à armes égales) ;
    cnn_10k    : petit CNN, référence = 10 000 images de test, N = 10 000 (protocole du projet).
Normalisations : ratio = FID / plancher ; z = (FID - plancher) / écart-type du plancher ;
part_baseline = (FID - plancher) / (FID de la baseline GAN - plancher), même réglage.
"""
import csv
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

import numpy as np
import torch
import torch.nn.functional as F

from audit_imbalance import between_class_share
from evaluate import fid as fid_reference

torch.set_num_threads(2)
_sqrt_cache = {}


def ref_sqrt(f_real):
    """(moyenne, covariance, racine carrée symétrique de la covariance) de la référence, calculées une fois."""
    key = id(f_real)
    if key not in _sqrt_cache:
        a = f_real.double().numpy()
        s = np.cov(a, rowvar=False)
        w, v = np.linalg.eigh(s)
        _sqrt_cache[key] = (a.mean(0), s, (v * np.sqrt(w.clip(0))) @ v.T, f_real)  # garde f_real en vie (id stable)
    return _sqrt_cache[key][:3]


def fid(f_real, f_fake):
    """Même valeur que evaluate.fid, en plus rapide (evaluate.fid : ~30 s par appel en dimension 2048 sur ce CPU).

    Tr((S_r S_g)^(1/2)) = somme des racines des valeurs propres de A S_g A, avec A = S_r^(1/2) ; et, si X est la
    matrice centrée des features générées, A S_g A = (XA)^T (XA) / (n-1) a les mêmes valeurs propres non nulles que
    (XA)(XA)^T / (n-1), de taille n x n. Vérifié contre evaluate.fid dans check_fast_fid.
    """
    mu1, s1, a = ref_sqrt(f_real)
    x = f_fake.double().numpy()
    mu2 = x.mean(0)
    xa = (x - mu2) @ a
    gram = xa @ xa.T if len(x) < x.shape[1] else xa.T @ xa
    tr_covmean = np.sqrt(np.linalg.eigvalsh(gram / (len(x) - 1)).clip(0)).sum()
    s2_trace = ((x - mu2) ** 2).sum() / (len(x) - 1)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + s2_trace - 2 * tr_covmean)


def check_fast_fid(f_real, f_fake):
    """Écart entre fid (rapide) et evaluate.fid (référence du dépôt)."""
    fast, slow = fid(f_real, f_fake), fid_reference(f_real, f_fake)
    print(f'controle fid rapide {fast:.4f} vs evaluate.fid {slow:.4f}', flush=True)
    return dict(fast=round(fast, 4), evaluate_fid=round(slow, 4))
SCRATCH = sys.argv[1]
CACHE = os.path.join(SCRATCH, 'results', 'fid', 'cache')
OUT = os.path.join(SCRATCH, 'results', 'fid')
DRAWS = 5
DEGRADATIONS = ['noise0.1', 'noise0.3', 'blur1.0', 'blend', 'erase10', 'thicken']
INC_DEGRADATIONS = ['noise0.3', 'blur1.0', 'blend', 'erase10']
N_SMALL = 500
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
# Parts des chiffres de la baseline commune, moyenne des 3 seeds (results/comparison_local.csv, lignes « Baseline commune »)
BASELINE_SHARES = np.array([1251.7, 2511.3, 155.7, 572.3, 886.3, 465.7, 789.3, 1599.0, 542.7, 1226.0])


def load(name):
    """Tenseur ou dict en cache."""
    return torch.load(os.path.join(CACHE, f'{name}.pt'))


def shares_with_one(share_one):
    """Parts des chiffres : share_one pour le « 1 », le reste réparti également."""
    s = np.full(10, (1 - share_one) / 9)
    s[1] = share_one
    return s


def draw_with_shares(y, shares, n, g):
    """Indices de n éléments de y dont les chiffres suivent `shares` (tirage sans remise)."""
    counts = np.round(n * shares / shares.sum()).astype(int)
    out = []
    for d in range(10):
        idx = (y == d).nonzero().squeeze(1)
        assert counts[d] <= len(idx), (d, counts[d], len(idx))
        out.append(idx[torch.randperm(len(idx), generator=g)[:counts[d]]])
    return torch.cat(out)


def mixed(clean, degraded, fraction):
    """Ensemble où la part `fraction` des images est remplacée par sa version dégradée (mêmes images)."""
    k = round(fraction * len(clean))
    return torch.cat([degraded[:k], clean[k:]])


class Setting:
    """Un réseau + une référence + une taille N ; garde le plancher pour normaliser."""

    def __init__(self, name, ref, pool, pool_y, n, clean_deg, degraded, gans):
        self.name, self.ref, self.pool, self.pool_y, self.n = name, ref, pool, pool_y, n
        self.clean_deg, self.degraded, self.gans = clean_deg, degraded, gans
        self.floor_mean = self.floor_sd = self.baseline = None


def floor_draws(s, g):
    """FID de N vraies images du réservoir (proportions naturelles) contre la référence, DRAWS tirages."""
    return [fid(s.ref, s.pool[torch.randperm(len(s.pool), generator=g)[:s.n]]) for _ in range(DRAWS)]


def row(s, experiment, condition, level, values):
    """Ligne normalisée du tableau."""
    m, sd = float(np.mean(values)), float(np.std(values))
    out = dict(setting=s.name, experiment=experiment, condition=condition, level=level, n=s.n,
               fid=round(m, 3), fid_sd=round(sd, 3), draws=len(values))
    if s.floor_mean is not None:
        out['ratio_floor'] = round(m / s.floor_mean, 2)
        out['z'] = round((m - s.floor_mean) / max(s.floor_sd, 1e-9), 1)
    if s.baseline is not None:
        out['part_baseline'] = round((m - s.floor_mean) / (s.baseline - s.floor_mean), 3)
    print(out, flush=True)
    return out


def run_setting(s):
    """Toutes les expériences pour un réglage."""
    g = torch.Generator().manual_seed(1)
    rows = []
    floor = floor_draws(s, g)
    s.floor_mean, s.floor_sd = float(np.mean(floor)), float(np.std(floor))
    s.baseline = fid(s.ref, s.gans['baseline'][:s.n])
    rows.append(row(s, 'plancher', 'reel vs test', 'pool', floor))
    for name, f in s.gans.items():
        rows.append(row(s, 'gan', name, '', [fid(s.ref, f[:s.n])]))
    for name in s.degraded:
        for frac in FRACTIONS:
            rows.append(row(s, 'pixels', name, frac, [fid(s.ref, mixed(s.clean_deg, s.degraded[name], frac))]))
    rng = np.random.default_rng(0)
    for k in (1, 2, 5):
        values = []
        for _ in range(DRAWS):
            shares = np.ones(10)
            shares[rng.choice(10, k, replace=False)] = 0
            values.append(fid(s.ref, s.pool[draw_with_shares(s.pool_y, shares, s.n, g)]))
        rows.append(row(s, 'classes', 'chiffres retires', k, values))
    for name, shares in (('30% de 1', shares_with_one(0.3)), ('20% de 1', shares_with_one(0.2)),
                         ('parts de la baseline', BASELINE_SHARES)):
        values = [fid(s.ref, s.pool[draw_with_shares(s.pool_y, shares, s.n, g)]) for _ in range(DRAWS)]
        rows.append(row(s, 'classes', 'desequilibre', name, values))
    return rows


def n_bias(name, ref_full, pool, gan, sizes, g):
    """FID réel-vs-réel et GAN-vs-réel quand la référence et l'ensemble mesuré ont tous deux N images."""
    rows = []
    for n in sizes:
        real = [fid(ref_full[torch.randperm(len(ref_full), generator=g)[:n]],
                    pool[torch.randperm(len(pool), generator=g)[:n]]) for _ in range(DRAWS)]
        fake = [fid(ref_full[torch.randperm(len(ref_full), generator=g)[:n]],
                    gan[torch.randperm(len(gan), generator=g)[:n]]) for _ in range(DRAWS)] if n <= len(gan) else None
        for cond, v in (('reel vs test', real), ('baseline GAN vs test', fake)):
            if v is None:
                continue
            rows.append(dict(setting=name, experiment='biais N', condition=cond, level=n, n=n,
                             fid=round(float(np.mean(v)), 3), fid_sd=round(float(np.std(v)), 3), draws=DRAWS))
            print(rows[-1], flush=True)
    return rows


def write(rows, filename):
    """CSV avec l'union des colonnes."""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, filename), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    lab = load('labels')
    y_test, y_train, y_val = lab['test'], lab['train'], lab['val']
    i_test, i_train, i_val = lab['i_test'], lab['i_train'], lab['i_val']
    cnn_test, cnn_train, cnn_val = load('cnn_test'), load('cnn_train'), load('cnn_val')

    # Réservoir commun (mêmes images pour Inception et cnn_1k) : 1 500 train + 1 500 val
    half = len(load('inc_pool_train'))
    pool_y = torch.cat([y_train[i_train[:half]], y_val[i_val[:half]]])
    inc_pool = torch.cat([load('inc_pool_train'), load('inc_pool_val')])
    cnn_pool = torch.cat([cnn_train[i_train[:half]], cnn_val[i_val[:half]]])
    inc = Setting('inception', load('inc_test_ref'), inc_pool, pool_y, N_SMALL, load('inc_pool_train')[:N_SMALL],
                  {d: load(f'inc_deg_{d}') for d in INC_DEGRADATIONS},
                  {'baseline': load('inc_gan_baseline'), 'baseline remise en proportion': load('inc_gan_baseline_rebalanced'),
                   'rpgan_r1_ema': load('inc_gan_rpgan')})
    cnn1 = Setting('cnn_500', cnn_test[i_test[:2000]], cnn_pool, pool_y, N_SMALL, cnn_train[i_train[:N_SMALL]],
                   {d: load(f'cnn_deg_{d}')[:N_SMALL] for d in DEGRADATIONS},
                   {'baseline': load('cnn_gan_cmploc_baseline_s0')[:N_SMALL],
                    'baseline remise en proportion': load('cnn_gan_baseline_rebalanced')[:N_SMALL],
                    'rpgan_r1_ema': load('cnn_gan_cmploc_rpgan_r1_ema_s0')[:N_SMALL]})
    # cnn_10k : réservoir = train hors des 10 000 images dégradées
    rest = i_train[10_000:]
    cnn10 = Setting('cnn_10k', cnn_test, cnn_train[rest], y_train[rest], 10_000, cnn_train[i_train[:10_000]],
                    {d: load(f'cnn_deg_{d}') for d in DEGRADATIONS},
                    {'baseline': load('cnn_gan_cmploc_baseline_s0')[:10_000],
                     'baseline remise en proportion': load('cnn_gan_baseline_rebalanced'),
                     'rpgan_r1_ema': load('cnn_gan_cmploc_rpgan_r1_ema_s0')[:10_000]})

    write([check_fast_fid(cnn_test, cnn_train[rest[:10_000]]), check_fast_fid(cnn1.ref, cnn1.gans['baseline']),
           check_fast_fid(inc.ref, inc.gans['baseline'])], 'fast_fid_check.csv')
    rows = []
    for s in (cnn10, cnn1, inc):
        rows += run_setting(s)
    # Planchers séparés train / val contre test, et train contre val (CNN, 10 000 images)
    for name, f in (('train 10k vs test', cnn_train[rest[:10_000]]), ('val 10k vs test', cnn_val)):
        rows.append(dict(setting='cnn_10k', experiment='plancher', condition=name, n=10_000, fid=round(fid(cnn_test, f), 3)))
    rows.append(dict(setting='cnn_10k', experiment='plancher', condition='train 10k vs val',
                     n=10_000, fid=round(fid(cnn_val, cnn_train[rest[:10_000]]), 3)))
    for name, inc_f, cnn_f in (('train 500 vs test', load('inc_pool_train')[half - N_SMALL:half], cnn_train[i_train[half - N_SMALL:half]]),
                               ('val 500 vs test', load('inc_pool_val')[:N_SMALL], cnn_val[i_val[:N_SMALL]])):
        rows.append(dict(setting='inception', experiment='plancher', condition=name, n=N_SMALL, fid=round(fid(inc.ref, inc_f), 3)))
        rows.append(dict(setting='cnn_500', experiment='plancher', condition=name, n=N_SMALL, fid=round(fid(cnn1.ref, cnn_f), 3)))
    g = torch.Generator().manual_seed(2)
    rows += n_bias('inception', inc.ref, inc_pool, torch.cat([inc.gans['baseline']]), [250, 500, 1000], g)
    rows += n_bias('cnn', cnn_test, cnn_train[rest], load('cnn_gan_cmploc_baseline_s0'),
                   [250, 500, 1000, 2000, 5000, 10_000], g)
    write(rows, 'audit.csv')

    # Part de la variance des features expliquée par le chiffre (mêmes 2 000 images de test)
    var_rows = [dict(network='inception', n=2000, between_class_share=round(between_class_share(inc.ref, y_test[i_test[:2000]]), 3)),
                dict(network='cnn', n=2000, between_class_share=round(between_class_share(cnn1.ref, y_test[i_test[:2000]]), 3)),
                dict(network='cnn', n=10_000, between_class_share=round(between_class_share(cnn_test, y_test), 3))]
    write(var_rows, 'variance_share.csv')
    print(var_rows)

    # Features avant / après ReLU (CNN, 10 000 images)
    relu_rows = []
    for name, tr in (('avant ReLU (evaluate.py)', lambda f: f), ('apres ReLU', F.relu)):
        t = tr(cnn_test)
        relu_rows.append(dict(features=name, dims_toujours_nulles=int((t.abs().max(0).values == 0).sum()),
                              rang_covariance=int(np.linalg.matrix_rank(np.cov(t.double().numpy(), rowvar=False))),
                              plancher_train_vs_test=round(fid(t, tr(cnn_train[rest[:10_000]])), 3),
                              baseline=round(fid(t, tr(cnn10.gans['baseline'])), 3),
                              rpgan=round(fid(t, tr(cnn10.gans['rpgan_r1_ema'])), 3)))
    write(relu_rows, 'relu.csv')
    print(relu_rows)
