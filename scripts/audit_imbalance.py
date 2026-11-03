"""Pourquoi un FID réagit-il au déséquilibre des chiffres ? Mesure sur de vraies images, sans aucun GAN.

Pour un réseau donné :
    - part de la variance des features expliquée par le chiffre (variance entre chiffres / variance totale) ;
    - FID entre 10 000 vraies images de train et val, avec les bonnes proportions (le plafond) ;
    - le même FID avec les proportions de l'époque 200 de la baseline, décomposé en terme de moyenne et de covariance.

Utilisation, depuis la racine du repo :
    python scripts/audit_imbalance.py cnn         # quelques secondes
    python scripts/audit_imbalance.py inception   # plusieurs minutes sur CPU
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from scipy import linalg

from data import mnist_split
from evaluate import inception_features, load_classifier, predict

DEVICE = torch.device('cpu')
N_POOL = 30_000
EPOCH_200_SHARES = np.array([12.3, 24.7, 2.3, 6.0, 8.2, 4.2, 6.7, 16.5, 5.3, 13.9])


def features(network, images):
    """Features (N, d) d'images (N, 1, 28, 28) dans [0, 1], avec le réseau demandé."""
    if network == 'cnn':
        return predict(load_classifier(DEVICE), images, DEVICE)[0]
    return inception_features(images, DEVICE)


def between_class_share(f, y):
    """Part de la variance totale expliquée par le chiffre : tr(S_between) / tr(S_total)."""
    f = f.double()
    grand_mean = f.mean(0)
    between = sum((y == c).double().mean() * ((f[y == c].mean(0) - grand_mean) ** 2).sum() for c in range(10))
    return (between / ((f - grand_mean) ** 2).sum(1).mean()).item()


def fid_terms(f_real, f_fake):
    """(terme de moyenne, terme de covariance) du FID."""
    a, b = f_real.double().numpy(), f_fake.double().numpy()
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(((a.mean(0) - b.mean(0)) ** 2).sum()), float(np.trace(s1 + s2 - 2 * linalg.sqrtm(s1 @ s2).real))


def subset_with_shares(y, shares, n=10_000):
    """Indices d'un sous-ensemble de n images dont les chiffres suivent les parts données (en %)."""
    counts = np.round(n * shares / shares.sum()).astype(int)
    return torch.cat([(y == c).nonzero().squeeze(1)[:counts[c]] for c in range(10)])


if __name__ == '__main__':
    network = sys.argv[1]
    x_val, y_val = mnist_split('val')
    x_pool, y_pool = mnist_split('train')
    x_pool, y_pool = x_pool[:N_POOL], y_pool[:N_POOL]
    f_val = features(network, x_val)
    f_pool = features(network, x_pool)
    real_shares = 100 * y_val.bincount(minlength=10).numpy() / len(y_val)
    print(network, 'part de variance due au chiffre', round(between_class_share(f_val, y_val), 3), flush=True)
    print(network, 'variance totale', round(((f_val.double() - f_val.double().mean(0)) ** 2).sum(1).mean().item(), 1), flush=True)
    for name, shares in (('bonnes proportions', real_shares), ('proportions de epoque 200', EPOCH_200_SHARES)):
        mean_term, cov_term = fid_terms(f_val, f_pool[subset_with_shares(y_pool, shares)])
        print(f'{network} {name}: FID {mean_term + cov_term:.2f} (moyenne {mean_term:.2f}, covariance {cov_term:.2f})', flush=True)
