"""Tests de sensibilité des mesures, à 10 000 images : FID et KID Inception, FID du petit CNN.

Pas de KID pour le CNN : ses features (avant ReLU) ont une échelle où le noyau polynomial du KID explose.

On prend de vraies images MNIST (split train), on leur applique un défaut connu et dosé, et on mesure chaque métrique
contre MNIST test (10 000 images). Mêmes images pour les deux réseaux. Refait à N = 10 000 l'audit CPU du 8/11
(Inception y était limité à 500 images) pour trancher la contradiction avec Lucic et al. 2018 (fig. 1b).
Pour comparer des métriques d'échelles différentes, chaque ligne donne aussi « part_baseline » :
(valeur − plancher) / (baseline GAN − plancher), dans la même métrique. 1 = ce défaut pèse autant que tout l'écart
entre la baseline et de vraies images.

Usage, depuis la racine du repo (branche noe/training), sur GPU :
    PYTHONPATH=. python scripts/sensitivity_10k.py <sortie.csv>
"""
import csv
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF

from data import mnist_split
from evaluate import fid, generate, inception_features, load_classifier, predict
from model import Generator
from scripts.kid_perclass import kid
from utils import get_device, load_model

N = 10_000
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
DRAWS = 3
GANS = {'baseline': 'runs/cmploc_baseline_s0/G.pth', 'rpgan_r1_ema': 'runs/cmploc_rpgan_r1_ema_s0/G.pth'}
# Parts des chiffres de la baseline commune, moyenne des 3 seeds (results/comparison_local.csv)
BASELINE_SHARES = np.array([1251.7, 2511.3, 155.7, 572.3, 886.3, 465.7, 789.3, 1599.0, 542.7, 1226.0])


# --- Dégradations : images (N, 1, 28, 28) dans [0, 1] -> même forme (mêmes définitions que l'audit CPU) ----------

def add_noise(x, sigma, seed=0):
    """Bruit gaussien additif d'écart-type sigma, puis écrêtage à [0, 1]."""
    return (x + sigma * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))).clamp(0, 1)


def blur(x, sigma):
    """Flou gaussien (noyau 5x5)."""
    return TF.gaussian_blur(x, kernel_size=5, sigma=sigma)


def blend(x, y, x_other, y_other, seed=0):
    """Chiffre « chimère » : moyenne de chaque image avec une image d'un autre chiffre."""
    g = torch.Generator().manual_seed(seed)
    partners = [(y_other != label).nonzero().squeeze(1) for label in y.tolist()]
    idx = [p[torch.randint(len(p), (1,), generator=g)].item() for p in partners]
    return 0.5 * x + 0.5 * x_other[idx]


def erase(x, size=10, seed=0):
    """Efface un carré size x size placé au hasard dans la zone centrale 20x20."""
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for i in range(len(x)):
        r, c = torch.randint(4, 24 - size + 1, (2,), generator=g).tolist()
        out[i, :, r:r + size, c:c + size] = 0
    return out


def thicken(x):
    """Épaissit le trait (dilatation 3x3)."""
    return torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)


def degradations(x, y, x_other, y_other):
    """Dictionnaire nom -> images dégradées."""
    return {'noise0.1': add_noise(x, 0.1), 'noise0.3': add_noise(x, 0.3), 'blur1.0': blur(x, 1.0),
            'blend': blend(x, y, x_other, y_other), 'erase10': erase(x), 'thicken': thicken(x)}


# --- Tirages par chiffre ------------------------------------------------------------------------------------

def draw_with_shares(y, shares, n, g):
    """Indices de n éléments de y dont les chiffres suivent `shares` (tirage sans remise)."""
    counts = np.round(n * shares / shares.sum()).astype(int)
    return torch.cat([(y == d).nonzero().squeeze(1)[torch.randperm(int((y == d).sum()), generator=g)[:counts[d]]]
                      for d in range(10)])


def shares_with_one(share_one):
    """Parts des chiffres : share_one pour le « 1 », le reste réparti également."""
    s = np.full(10, (1 - share_one) / 9)
    s[1] = share_one
    return s


def rebalance(images, preds, labels, n):
    """n images dont les chiffres prédits suivent les proportions de `labels`."""
    counts = (torch.bincount(labels, minlength=10).double() / len(labels) * n).round().long()
    return torch.cat([images[preds == d][:counts[d]] for d in range(10)])


# --- Mesures ------------------------------------------------------------------------------------------------

class Features:
    """Features Inception et CNN d'un ensemble d'images, calculées une fois."""

    def __init__(self, images, device, clf):
        self.inc = inception_features(images, device)
        self.cnn = predict(clf, images, device)[0]

    def subset(self, idx):
        """Mêmes features restreintes aux indices idx."""
        out = object.__new__(Features)
        out.inc, out.cnn = self.inc[idx], self.cnn[idx]
        return out


def metrics(ref, f):
    """FID et KID (× 1000) Inception, FID du CNN, de f contre ref."""
    return dict(fid_inception=fid(ref.inc, f.inc), kid_inception=1000 * kid(ref.inc, f.inc)[0],
                fid_cnn=fid(ref.cnn, f.cnn))


def mixed(clean, degraded, fraction):
    """Ensemble où la part `fraction` des images est remplacée par sa version dégradée (mêmes images)."""
    k = round(fraction * N)
    out = object.__new__(Features)
    out.inc = torch.cat([degraded.inc[:k], clean.inc[k:]])
    out.cnn = torch.cat([degraded.cnn[:k], clean.cnn[k:]])
    return out


def mean_metrics(values):
    """Moyenne et écart-type, clé par clé, d'une liste de dictionnaires de métriques."""
    keys = values[0].keys()
    return ({k: float(np.mean([v[k] for v in values])) for k in keys},
            {k: float(np.std([v[k] for v in values])) for k in keys})


def conditions(device, clf):
    """Génère (expérience, condition, niveau, liste de dictionnaires de métriques) pour toutes les conditions."""
    g = torch.Generator().manual_seed(0)
    x_test, _ = mnist_split('test')
    x_train, y_train = mnist_split('train')
    ref = Features(x_test, device, clf)
    pool = Features(x_train, device, clf)
    perm = torch.randperm(len(x_train), generator=g)
    xd, yd = x_train[perm[:N]], y_train[perm[:N]]
    xo, yo = x_train[perm[-5000:]], y_train[perm[-5000:]]
    clean = pool.subset(perm[:N])

    yield 'plancher', 'reel train vs test', '', [metrics(ref, pool.subset(torch.randperm(len(x_train), generator=g)[:N]))
                                                 for _ in range(DRAWS)]
    for name, path in GANS.items():
        G = load_model(Generator(784).to(device), path, device).eval()
        big = generate(G, 60_000, device, seed=0)
        yield 'gan', name, '', [metrics(ref, Features(big[:N], device, clf))]
        if name == 'baseline':
            bal = rebalance(big, predict(clf, big, device)[1], mnist_split('test')[1], N)
            yield 'gan', 'baseline remise en proportion', '', [metrics(ref, Features(bal, device, clf))]
    for name, x in degradations(xd, yd, xo, yo).items():
        degraded = Features(x, device, clf)
        for frac in FRACTIONS:
            yield 'pixels', name, frac, [metrics(ref, mixed(clean, degraded, frac))]
    rng = np.random.default_rng(0)
    for k in (1, 2, 5):
        values = []
        for _ in range(DRAWS):
            shares = np.ones(10)
            shares[rng.choice(10, k, replace=False)] = 0
            values.append(metrics(ref, pool.subset(draw_with_shares(y_train, shares, N, g))))
        yield 'classes', 'chiffres retires', k, values
    for name, shares in (('30% de 1', shares_with_one(0.3)), ('20% de 1', shares_with_one(0.2)),
                         ('parts de la baseline', BASELINE_SHARES)):
        yield 'classes', 'desequilibre', name, [metrics(ref, pool.subset(draw_with_shares(y_train, shares, N, g)))
                                                for _ in range(DRAWS)]


if __name__ == '__main__':
    device = get_device()
    clf = load_classifier(device)
    keys = ('fid_inception', 'kid_inception', 'fid_cnn')
    rows, floor, base = [], None, None
    for experiment, condition, level, values in conditions(device, clf):
        m, sd = mean_metrics(values)
        floor = m if experiment == 'plancher' else floor
        base = m if (experiment, condition) == ('gan', 'baseline') else base
        row = dict(experiment=experiment, condition=condition, level=level, draws=len(values),
                   **{k: m[k] for k in keys}, **{f'{k}_sd': sd[k] for k in keys})
        if base is not None:
            row.update({f'{k}_part_baseline': (m[k] - floor[k]) / (base[k] - floor[k]) for k in keys})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(sys.argv[1], 'w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        out.writeheader()
        out.writerows(rows)
