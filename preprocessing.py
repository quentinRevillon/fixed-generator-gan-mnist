"""Options du volet Données, activées séparément dans le YAML (section data). Détails : rapport-donnée-illustré.md.

- selection : Instance Selection for GANs (DeVries et al., 2020), entraîner sur les images les plus denses.
- instance_noise : bruit gaussien sur les entrées de D, réelles et générées (Sønderby et al., 2017).

Code de Nicodème, adapté au split commun de data.py. Utilisation (voir train.py) :
    dataset = training_set(train_dataset(), cfg, device)
    D_input = add_instance_noise(D, cfg)
"""
import torch
import torch.nn as nn

from data import mnist_split
from evaluate import load_classifier, predict

PCA_DIM = 50


def embed(x, space, device):
    """Représentation dans laquelle on estime la densité : features du classifieur ou ACP des pixels."""
    if space == 'classifier':
        return predict(load_classifier(device), x, device)[0]
    if space == 'pixels':
        f = x.flatten(1).double()
        f = f - f.mean(0)
        return f @ torch.linalg.eigh(f.T @ f).eigenvectors[:, -PCA_DIM:]
    raise ValueError(f'Espace inconnu : {space}')


def gaussian_score(f):
    """Log-vraisemblance d'une gaussienne ajustée sur f, à une constante près (- distance de Mahalanobis²)."""
    d = f.double() - f.double().mean(0)
    cov = d.T @ d / (len(d) - 1) + 1e-6 * torch.eye(d.shape[1], dtype=d.dtype)
    return -(d * torch.linalg.solve(cov, d.T).T).sum(1)


def top(scores, idx, keep):
    """Les indices de la fraction keep des meilleurs scores."""
    return idx[scores.topk(round(keep * len(idx))).indices]


def select_instances(y, f, keep, per_class=True):
    """Indices des images gardées : la fraction keep la plus dense, dans chaque classe ou globalement."""
    if not per_class:
        return top(gaussian_score(f), torch.arange(len(y)), keep).sort().values
    groups = [(y == c).nonzero().squeeze(1) for c in y.unique()]
    return torch.cat([top(gaussian_score(f[g]), g, keep) for g in groups]).sort().values


def training_set(dataset, cfg, device):
    """Applique data.selection au dataset ; sans la section, ou avec enabled: false, il ressort inchangé."""
    sel = cfg.get('data', {}).get('selection') or {}
    if not sel.get('enabled'):
        return dataset
    x, y = mnist_split('train')
    idx = select_instances(y, embed(x, sel.get('space', 'classifier'), device), sel['keep'],
                           sel.get('per_class', True))
    print('Sélection : ' + ' '.join(f'{c}={n}' for c, n in enumerate(torch.bincount(y[idx]).tolist())))
    return torch.utils.data.Subset(dataset, idx.tolist())


def noise_sigma(cfg, epoch):
    """Écart-type de data.instance_noise : décroissance linéaire vers 0 (constant si anneal_epochs vaut 0)."""
    noise = cfg.get('data', {}).get('instance_noise') or {}
    if not noise.get('enabled'):
        return 0.
    sigma, anneal = noise['sigma'], noise.get('anneal_epochs', 0)
    return sigma * max(0., 1 - (epoch - 1) / anneal) if anneal else sigma


def add_instance_noise(D, cfg):
    """D enveloppé par InstanceNoise si data.instance_noise est activé, sinon D lui-même."""
    return InstanceNoise(D) if (cfg.get('data', {}).get('instance_noise') or {}).get('enabled') else D


class InstanceNoise(nn.Module):
    """Enveloppe D et bruite ses entrées (dans [-1, 1]). À sigma = 0, aucun tirage : la baseline reste identique."""

    def __init__(self, D):
        super().__init__()
        self.D, self.sigma = D, 0.

    def forward(self, x):
        return self.D(x + self.sigma * torch.randn_like(x) if self.sigma else x)
