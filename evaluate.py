"""Évaluation : FID, precision / recall (k-NN), histogramme des classes, tableaux de résultats.

Deux mesures, jamais mélangées :
    'cnn'  : features d'un petit classifieur MNIST partagé (checkpoints/mnist_classifier.pth), k = 3 -> results/results.csv
    'prof' : FID sur Inception, precision / recall sur VGG (k = 5, 9 984 images), comme le classement -> results/results_prof.csv
Les features sont calculées sur une référence réelle (split 'val' pendant le développement, 'test' pour les
chiffres finaux, voir data.py) et sur les images générées. Détails et formules : architecture-choix.md.

En ligne de commande (chaque évaluation d'un run ajoute une ligne par seed au tableau de sa mesure) :
    python evaluate.py --run runs/baseline_200ep_prof --weights best --seeds 0 1 2            # plain, petit CNN
    python evaluate.py --run runs/baseline_200ep_prof --weights best --seeds 0 1 2 --metric prof
    python evaluate.py --run checkpoints --sampling configs/sampling/plain.yaml --reference test
    python evaluate.py --samples samples         # PNG produits par generate.py (comme le correcteur)
    python evaluate.py --sanity                  # plafond : 10 000 images réelles du split train vs référence
    python evaluate.py --train-classifier        # (ré)entraîne le classifieur de features
    python evaluate.py --run checkpoints --metric prof --reference test     # réseaux de la plateforme du prof

Depuis Python :
    from evaluate import evaluate, generate
    m = evaluate(images, device)                     # images : (N, 1, 28, 28) dans [0, 1] ; référence 'val'
    m['fid'], m['precision'], m['recall'], m['classes']
"""
import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy import linalg
from torchvision import transforms

from data import mnist_raw, mnist_split
from model import Discriminator, Generator
from sampling import sample
from utils import get_device, git_commit, load_config, load_model, set_seed

CLF_PATH = 'checkpoints/mnist_classifier.pth'


# --- Métriques -------------------------------------------------------------------------------------------------

class Classifier(nn.Module):
    """Petit CNN MNIST ; sa couche de 128 dimensions sert d'espace de features pour les métriques."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.out = nn.Linear(128, 10)

    def features(self, x):  # x : (N, 1, 28, 28) dans [0, 1] ; avant ReLU (sinon covariance singulière)
        return self.fc(self.conv(x).flatten(1))

    def forward(self, x):
        return self.out(F.relu(self.features(x)))


def train_classifier(device, epochs=3):
    """Entraîne le classifieur de features sur les 60 000 images de MNIST train et le sauvegarde dans CLF_PATH.

    Il voit aussi les images de validation : acceptable, car il ne sert qu'à définir l'espace de features,
    identique pour toutes les méthodes comparées (aucun choix n'est fait avec lui).
    """
    set_seed(0)
    x, y = mnist_raw(train=True)
    x = x.unsqueeze(1).float() / 255
    clf = Classifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    xt, yt = mnist_split('test')
    acc = (predict(clf, xt, device)[1] == yt).float().mean().item()
    print(f'Classifieur entraîné : précision test {acc:.4f} -> {CLF_PATH}')
    torch.save(clf.state_dict(), CLF_PATH)
    return clf


def load_classifier(device):
    """Charge le classifieur partagé (l'entraîne une fois s'il est absent)."""
    if not os.path.exists(CLF_PATH):
        print(f'{CLF_PATH} absent : entraînement du classifieur (une seule fois).')
        return train_classifier(device).eval()
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load(CLF_PATH, map_location=device))
    return clf.eval()


@torch.no_grad()
def predict(clf, x, device, bs=1000):
    """Renvoie (features, classes prédites) sur le CPU."""
    feats, preds = [], []
    for i in range(0, len(x), bs):
        f = clf.features(x[i:i + bs].to(device))
        feats.append(f.cpu())
        preds.append(clf.out(F.relu(f)).argmax(1).cpu())
    return torch.cat(feats), torch.cat(preds)


def fid(f_real, f_fake):
    """FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))."""
    f1, f2 = f_real.double().numpy(), f_fake.double().numpy()
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def knn_radii(f, k, bs=1000):
    """Distance de chaque point à son k-ième plus proche voisin (lui-même exclu)."""
    return torch.cat([torch.cdist(f[i:i + bs], f).topk(k + 1, largest=False).values[:, -1]
                      for i in range(0, len(f), bs)])


def coverage(a, b, radii_b, bs=1000):
    """Proportion des points de a qui tombent dans au moins une boule B(b_j, radii_b[j])."""
    inside = [(torch.cdist(a[i:i + bs], b) <= radii_b[None]).any(1) for i in range(0, len(a), bs)]
    return torch.cat(inside).float().mean().item()


_real_cache = {}


def real_features(clf, device, reference):
    """Features du split réel de référence ('val' ou 'test'), calculées une fois puis mises en cache."""
    if reference not in _real_cache:
        _real_cache[reference] = predict(clf, mnist_split(reference)[0], device)[0]
    return _real_cache[reference]


def evaluate(images, device, k=3, reference='val'):
    """images : (N, 1, 28, 28) dans [0, 1]. Compare au split réel `reference` ('val' ou 'test')."""
    clf = load_classifier(device)
    f_real = real_features(clf, device, reference)
    f_fake, preds = predict(clf, images, device)
    return dict(fid=fid(f_real, f_fake),
                precision=coverage(f_fake, f_real, knn_radii(f_real, k)),  # qualité
                recall=coverage(f_real, f_fake, knn_radii(f_fake, k)),     # diversité
                classes=torch.bincount(preds, minlength=10).tolist(),
                reference=reference)


def to_unit_images(x):
    """Sorties de G (N, 784) dans [-1, 1] -> images (N, 1, 28, 28) dans [0, 1] (inverse de Normalize(0.5, 0.5))."""
    return ((x.cpu() + 1) / 2).view(-1, 1, 28, 28)


@torch.no_grad()
def generate(G, n, device, seed=0, bs=1000):
    """n images de G dans [0, 1], forme (n, 1, 28, 28).

    z est tiré sur CPU avec un générateur dédié (seed) : l'évaluation ne consomme pas le RNG global de
    l'entraînement, et les mêmes z sont utilisés quel que soit le device.
    """
    was_training = G.training
    G.eval()
    rng = torch.Generator().manual_seed(seed)
    imgs = torch.cat([G(torch.randn(min(bs, n - i), 100, generator=rng).to(device)).cpu()
                      for i in range(0, n, bs)])
    G.train(was_training)
    return to_unit_images(imgs)


def load_samples(folder):
    """Charge les PNG d'un dossier (ex. samples/) en tenseur (N, 1, 28, 28) dans [0, 1]."""
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(f).convert('L')) for f in files])


def format_metrics(m):
    """Résumé lisible d'un dict renvoyé par evaluate : FID / P / R et part de chaque chiffre."""
    total = sum(m['classes'])
    shares = ' '.join(f'{100 * c / total:.1f}' for c in m['classes'])
    return (f"ref={m['reference']} FID={m['fid']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f}\n"
            f'Classes 0-9 (%) : {shares}')


def save_grid(images, path):
    """Grille 10x10 des 100 premières images (N, 1, 28, 28) dans [0, 1]."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torchvision.utils.save_image(images[:100], path, nrow=10)


# --- Mesure de la plateforme du prof : FID Inception, precision / recall VGG (k = 5, 9 984 images) ------------
# Page de résultats : "Historical FID uses Inception features on all images; historical P/R uses VGG features
# with k=5 and 9,984 images". Prétraitement, poids et référence n'y sont pas précisés : c'est une approximation.

PROF_K, PROF_PR_IMAGES = 5, 9984
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_prof_cache = {}


@lru_cache(maxsize=None)
def load_inception(device):
    """Inception-v3 aux poids standard du FID (pytorch-fid), dernière couche de pooling, 2048 dimensions."""
    from pytorch_fid.inception import InceptionV3  # import local : generate.py n'en a pas besoin
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()


@lru_cache(maxsize=None)
def load_vgg(device):
    """VGG16 ImageNet coupé après la 2e couche entièrement connectée, 4096 dimensions (Kynkäänniemi et al. 2019)."""
    vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    return nn.Sequential(vgg.features, vgg.avgpool, nn.Flatten(), *list(vgg.classifier)[:4]).to(device).eval()


@torch.no_grad()
def inception_features(images, device, bs=256):
    """Features (N, 2048) d'images (N, 1, 28, 28) dans [0, 1] ; le réseau redimensionne et normalise lui-même."""
    net = load_inception(device)
    return torch.cat([net(images[i:i + bs].to(device).repeat(1, 3, 1, 1))[0].flatten(1).cpu()
                      for i in range(0, len(images), bs)])


def vgg_preprocess(x, device):
    """Images (N, 1, 28, 28) dans [0, 1] -> (N, 3, 224, 224) normalisées comme ImageNet."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = F.interpolate(x.to(device).repeat(1, 3, 1, 1), size=224, mode='bilinear', align_corners=False)
    return (x - mean) / std


@torch.no_grad()
def vgg_features(images, device, bs=256):
    """Features (N, 4096) d'images (N, 1, 28, 28) dans [0, 1]."""
    net = load_vgg(device)
    return torch.cat([net(vgg_preprocess(images[i:i + bs], device)).cpu() for i in range(0, len(images), bs)])


def prof_real_features(device, reference):
    """(features Inception, features VGG) du split réel `reference`, calculées une fois puis mises en cache."""
    if reference not in _prof_cache:
        real = mnist_split(reference)[0]
        _prof_cache[reference] = inception_features(real, device), vgg_features(real[:PROF_PR_IMAGES], device)
    return _prof_cache[reference]


def evaluate_prof(images, device, reference='val'):
    """Comme evaluate, mais avec les réseaux de la plateforme ; les classes viennent toujours du petit CNN."""
    f_real, v_real = prof_real_features(device, reference)
    v_fake = vgg_features(images[:PROF_PR_IMAGES], device)
    return dict(fid=fid(f_real, inception_features(images, device)),
                precision=coverage(v_fake, v_real, knn_radii(v_real, PROF_K)),
                recall=coverage(v_real, v_fake, knn_radii(v_fake, PROF_K)),
                classes=predict(load_classifier(device), images, device)[1].bincount(minlength=10).tolist(),
                reference=reference)


EVALUATORS = {'cnn': evaluate, 'prof': evaluate_prof}


# --- Tableau commun des résultats : results/results.csv, une ligne par évaluation ------------------------------
# Colonnes : date, git_commit | run, weights ('last' = G.pth, 'best' = G_best.pth) | reference ('val' ou 'test')
# | method, params (JSON) | seed, n | fid, precision, recall | acceptance_rate, n_proposals, seconds (coût,
# slide 82) | classes (nombre d'images par chiffre 0-9, séparés par '-').

RESULTS_CSV = {'cnn': 'results/results.csv', 'prof': 'results/results_prof.csv'}  # deux mesures, jamais mélangées
COLUMNS = ['date', 'git_commit', 'run', 'weights', 'reference', 'method', 'params', 'seed', 'n',
           'fid', 'precision', 'recall', 'acceptance_rate', 'n_proposals', 'seconds', 'classes']
SUMMARY_METRICS = ['fid', 'precision', 'recall', 'acceptance_rate', 'seconds']


def make_row(run, weights, reference, sampling_cfg, seed, metrics, stats):
    """Construit une ligne du tableau à partir de la config d'échantillonnage, des métriques et des stats."""
    params = {k: v for k, v in sampling_cfg.items() if k != 'method'}
    return {'date': time.strftime('%Y-%m-%d %H:%M'), 'git_commit': git_commit(), 'run': run, 'weights': weights,
            'reference': reference, 'method': sampling_cfg['method'], 'params': json.dumps(params, sort_keys=True),
            'seed': seed, 'n': sum(metrics['classes']), 'fid': round(metrics['fid'], 4),
            'precision': round(metrics['precision'], 4), 'recall': round(metrics['recall'], 4),
            'acceptance_rate': round(stats['acceptance_rate'], 4), 'n_proposals': stats['n_proposals'],
            'seconds': round(stats['seconds'], 3), 'classes': '-'.join(map(str, metrics['classes']))}


def append_result(row, metric='cnn'):
    """Ajoute une ligne (dict) au CSV de la mesure `metric`, en créant le fichier et son en-tête si besoin."""
    path = RESULTS_CSV[metric]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def load_results(metric='cnn'):
    """Toutes les lignes du CSV de la mesure `metric` (liste de dicts, valeurs en texte)."""
    with open(RESULTS_CSV[metric], newline='') as f:
        return list(csv.DictReader(f))


def summarize(rows):
    """Moyenne et écart-type des métriques par (run, weights, reference, method, params), toutes seeds confondues."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['run'], r['weights'], r['reference'], r['method'], r['params'])].append(r)
    out = {}
    for key, group in groups.items():
        agg = {'n_seeds': len(group)}
        for m in SUMMARY_METRICS:
            values = np.array([float(r[m]) for r in group])
            agg[f'{m}_mean'], agg[f'{m}_std'] = values.mean(), values.std()
        out[key] = agg
    return out


def format_summary(key, agg):
    """Une ligne lisible : méthode, paramètres, FID / P / R (moyenne ± écart-type), acceptation, durée."""
    run, weights, reference, method, params = key
    pm = lambda m, d: f"{agg[f'{m}_mean']:.{d}f}±{agg[f'{m}_std']:.{d}f}"
    return (f"{method:10s} {params:28s} ref={reference:4s} seeds={agg['n_seeds']} | FID={pm('fid', 1)} "
            f"P={pm('precision', 3)} R={pm('recall', 3)} | accept={agg['acceptance_rate_mean']:.3f} "
            f"t={agg['seconds_mean']:.1f}s")


# --- Évaluer une méthode d'échantillonnage sur un run (utilisé par la CLI ci-dessous et par sweep.py) ----------

WEIGHT_FILES = {'last': ('G.pth', 'D.pth'), 'best': ('G_best.pth', 'D_best.pth'),
                'best_kt': ('G_best.pth', 'D_best_kt.pth')}  # best_kt : D passé par keep_training.py


def load_run(run_dir, weights, device):
    """(G, D) d'un dossier de run (ou de checkpoints/) : 'last' -> G.pth / D.pth, 'best' -> G_best.pth / D_best.pth."""
    g_file, d_file = WEIGHT_FILES[weights]
    G = load_model(Generator(g_output_dim=784).to(device), os.path.join(run_dir, g_file), device)
    D = load_model(Discriminator(784).to(device), os.path.join(run_dir, d_file), device)
    return G.eval(), D.eval()


def evaluate_sampling(G, D, sampling_cfg, seed, reference, device, run, weights, metric='cnn', n=10_000,
                      batch_size=10_000):
    """Tire n images avec sampling_cfg, les évalue contre `reference` ; renvoie (ligne du tableau, images)."""
    x, stats = sample(G, D, n, sampling_cfg, seed=seed, batch_size=batch_size)
    images = to_unit_images(x)
    metrics = EVALUATORS[metric](images, device, reference=reference)
    return make_row(run, weights, reference, sampling_cfg, seed, metrics, stats), images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a GAN generator and its sampling method on MNIST.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--run', help='Dossier contenant G.pth / D.pth (runs/<name> ou checkpoints).')
    src.add_argument('--samples', help='Dossier de PNG produits par generate.py.')
    src.add_argument('--sanity', action='store_true', help='Plafond : 10 000 images du split train vs référence.')
    src.add_argument('--train-classifier', action='store_true', help='(Ré)entraîne le classifieur de features.')
    parser.add_argument('--weights', choices=sorted(WEIGHT_FILES), default='last', help='Jeu de poids G et D du run.')
    parser.add_argument('--sampling', default='configs/sampling/plain.yaml', help='Méthode d\'échantillonnage (YAML).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Seeds de l\'échantillonnage.')
    parser.add_argument('--reference', choices=['train', 'val', 'test'], default='val', help='Split réel de référence.')
    parser.add_argument('--metric', choices=sorted(EVALUATORS), default='cnn', help='cnn : petit CNN ; prof : Inception et VGG.')
    args = parser.parse_args()
    device = get_device()

    if args.train_classifier:
        train_classifier(device)
    elif args.run:
        G, D = load_run(args.run, args.weights, device)
        cfg = load_config(args.sampling)
        rows = []
        for seed in args.seeds:
            row, images = evaluate_sampling(G, D, cfg, seed, args.reference, device, args.run, args.weights, args.metric)
            append_result(row, args.metric)
            rows.append(row)
            print(f"seed {seed} : FID={row['fid']:.2f} P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"accept={row['acceptance_rate']:.3f} t={row['seconds']:.1f}s")
        save_grid(images, os.path.join('figures', f"samples_{os.path.basename(args.run)}_{cfg['method']}.png"))
        for key, agg in summarize(rows).items():
            print(format_summary(key, agg))
    else:
        images = load_samples(args.samples) if args.samples else mnist_split('train')[0][:10_000]
        print(format_metrics(EVALUATORS[args.metric](images, device, reference=args.reference)))
