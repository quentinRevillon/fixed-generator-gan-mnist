"""Évaluation d'un générateur : FID, precision / recall (k-NN), grille d'images et histogramme des classes.

Les métriques comparent des features (avant-dernière couche, 128 dim) d'un petit classifieur MNIST
partagé par tout le groupe (checkpoints/mnist_classifier.pth), calculées sur le MNIST test (référence)
et sur les images générées. Détails et formules : architecture-choix.md.
"""
import argparse
import csv
import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy import linalg
from torchvision import datasets, transforms

from model import Generator
from utils import get_device, load_model, set_seed

CLF_PATH = 'checkpoints/mnist_classifier.pth'
RESULTS_CSV = 'results.csv'


class Classifier(nn.Module):
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


def mnist(train):
    """Images MNIST dans [0, 1], forme (N, 1, 28, 28), et labels."""
    path = os.getenv('DATA', 'data')
    ds = datasets.MNIST(root=path, train=train, download='DATA' not in os.environ)
    return ds.data.unsqueeze(1).float() / 255, ds.targets


def train_classifier(device, epochs=3):
    set_seed(0)
    x, y = mnist(train=True)
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
    xt, yt = mnist(train=False)
    acc = (predict(clf, xt, device)[1] == yt).float().mean().item()
    print(f'Classifieur entraîné : précision test {acc:.4f} -> {CLF_PATH}')
    torch.save(clf.state_dict(), CLF_PATH)
    return clf


def load_classifier(device):
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


def evaluate(images, device, k=3):
    """images : (N, 1, 28, 28) dans [0, 1]. Compare au MNIST test."""
    clf = load_classifier(device)
    if 'feats' not in _real_cache:
        _real_cache['feats'] = predict(clf, mnist(train=False)[0], device)[0]
    f_real = _real_cache['feats']
    f_fake, preds = predict(clf, images, device)
    return dict(fid=fid(f_real, f_fake),
                precision=coverage(f_fake, f_real, knn_radii(f_real, k)),  # qualité
                recall=coverage(f_real, f_fake, knn_radii(f_fake, k)),     # diversité
                classes=torch.bincount(preds, minlength=10).tolist())


@torch.no_grad()
def generate(G, n, device, bs=1000):
    was_training = G.training
    G.eval()
    imgs = torch.cat([(G(torch.randn(min(bs, n - i), 100, device=device)).cpu() + 1) / 2
                      for i in range(0, n, bs)])
    G.train(was_training)
    return imgs.view(-1, 1, 28, 28)


def load_samples(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(f).convert('L')) for f in files])


def report(name, m, images, grid_path):
    torchvision.utils.save_image(images[:100], grid_path, nrow=10)
    total = sum(m['classes'])
    print(f"{name} : FID={m['fid']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f}")
    print('Classes 0-9 (%) : ' + ' '.join(f'{100 * c / total:.1f}' for c in m['classes']))
    print(f'Grille : {grid_path}')
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['date', 'run', 'n', 'fid', 'precision', 'recall', 'classes'])
        w.writerow([time.strftime('%Y-%m-%d %H:%M'), name, len(images), f"{m['fid']:.4f}",
                    f"{m['precision']:.4f}", f"{m['recall']:.4f}", '-'.join(map(str, m['classes']))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a GAN generator on MNIST.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', help='Poids du générateur (ex. runs/baseline/G.pth).')
    src.add_argument('--samples', help='Dossier de PNG produits par generate.py.')
    src.add_argument('--sanity', action='store_true', help='Contrôle : MNIST train vs MNIST test.')
    src.add_argument('--train-classifier', action='store_true', help='(Ré)entraîne le classifieur de features.')
    parser.add_argument('--n', type=int, default=10000, help='Nombre d\'images générées.')
    parser.add_argument('--k', type=int, default=3, help='k plus proches voisins pour precision / recall.')
    args = parser.parse_args()
    device = get_device()

    if args.train_classifier:
        train_classifier(device)
    else:
        if args.checkpoint:
            set_seed(0)
            G = load_model(Generator(g_output_dim=784).to(device), args.checkpoint, device)
            images, name = generate(G, args.n, device), args.checkpoint
            grid_path = os.path.join(os.path.dirname(args.checkpoint) or '.', 'grid.png')
        elif args.samples:
            images, name, grid_path = load_samples(args.samples), args.samples, 'grid_samples.png'
        else:
            images, name, grid_path = mnist(train=True)[0][:args.n], 'sanity_train_vs_test', 'grid_sanity.png'
        report(name, evaluate(images, device, args.k), images, grid_path)
