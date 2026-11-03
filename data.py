"""Données MNIST et split figé train / val / test, commun à tout le groupe.

Split (contrat partagé, ne pas modifier sans prévenir le groupe) :
    - les 60 000 images de MNIST train sont permutées une fois avec SPLIT_SEED ;
    - les VAL_SIZE premières de la permutation forment la validation, les 50 000 autres l'entraînement ;
    - MNIST test (10 000) ne sert qu'aux chiffres finaux du rapport, jamais à un choix.

    split     | taille | sert à
    ----------|--------|------------------------------------------------------------
    'train'   | 50 000 | entraîner G et D
    'val'     | 10 000 | référence des métriques pendant l'entraînement, Optuna, choix de K / epoch
    'test'    | 10 000 | chiffres finaux (rapport, slides)

Utilisation :
    from data import mnist_split, train_loader
    x, y = mnist_split('val')                       # (10000, 1, 28, 28) dans [0, 1], labels
    loader = train_loader(batch_size=100, seed=0)   # batchs (x, y), x : (B, 784) dans [-1, 1]
"""
import os

import torch
from torchvision import datasets

SPLIT_SEED = 0
VAL_SIZE = 10_000


def mnist_raw(train):
    """Images MNIST brutes (uint8, (N, 28, 28)) et labels ; $DATA sur Juliet, sinon téléchargement dans data/."""
    ds = datasets.MNIST(root=os.getenv('DATA', 'data'), train=train, download='DATA' not in os.environ)
    return ds.data, ds.targets


def split_indices(n=60_000, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Indices (train, val) d'une permutation fixe de range(n)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[val_size:], perm[:val_size]


def mnist_split(split):
    """Images (N, 1, 28, 28) dans [0, 1] et labels du split 'train', 'val' ou 'test'."""
    if split == 'test':
        x, y = mnist_raw(train=False)
    else:
        x, y = mnist_raw(train=True)
        idx = dict(zip(('train', 'val'), split_indices()))[split]
        x, y = x[idx], y[idx]
    return x.unsqueeze(1).float() / 255, y


def train_loader(batch_size, seed):
    """DataLoader mélangé (ordre fixé par seed) sur le split 'train', images aplaties dans [-1, 1]."""
    x, y = mnist_split('train')
    dataset = torch.utils.data.TensorDataset(x.view(len(x), -1) * 2 - 1, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed))
