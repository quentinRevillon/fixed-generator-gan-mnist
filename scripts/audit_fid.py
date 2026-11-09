"""À quoi réagit chaque FID ? Mêmes images, avec les chiffres tels que générés puis remis aux proportions de val.

Mesure le FID du petit CNN et le FID Inception aux époques 120 et 200, contre val.

Utilisation, depuis la racine du repo :
    python scripts/audit_fid.py <dossier contenant G_ep120.pth et G_ep200.pth>
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch

from data import mnist_split
from evaluate import fid, generate, inception_features, load_classifier, predict
from model import Generator
from utils import load_model

DEVICE = torch.device('cpu')
POOL = 100_000


def rebalanced(images, preds, labels):
    """Sous-échantillon qui a exactement la répartition des chiffres de `labels`."""
    wanted = torch.bincount(labels, minlength=10).tolist()
    return torch.cat([images[preds == digit][:count] for digit, count in enumerate(wanted)])


def shares(preds):
    """Part de chaque chiffre, en %."""
    return [round(100 * v, 1) for v in (preds.bincount(minlength=10) / len(preds)).tolist()]


if __name__ == '__main__':
    clf = load_classifier(DEVICE)
    real, labels = mnist_split('val')
    cnn_real = predict(clf, real, DEVICE)[0]
    inception_real = inception_features(real, DEVICE)
    print('reels prets', flush=True)
    for epoch in (120, 200):
        G = load_model(Generator(784), os.path.join(sys.argv[1], f'G_ep{epoch}.pth'), 'cpu').eval()
        images = generate(G, POOL, DEVICE)
        preds = predict(clf, images, DEVICE)[1]
        raw, balanced = images[:10_000], rebalanced(images, preds, labels)
        assert len(balanced) == 10_000
        print(epoch, 'parts', shares(preds[:10_000]), flush=True)
        for name, x in (('tel quel', raw), ('remis en proportion', balanced)):
            cnn = fid(cnn_real, predict(clf, x, DEVICE)[0])
            inception = fid(inception_real, inception_features(x, DEVICE))
            print(f'epoque {epoch} {name}: FID CNN {cnn:.1f} FID Inception {inception:.1f}', flush=True)
