"""Audit du petit CNN : précision sur test, confiance sur les images générées, stabilité du FID selon la seed du CNN.

Ne modifie aucun fichier du dépôt. Lancer depuis la racine du dépôt :
    python scripts/audit_cnn.py <dossier contenant G_ep120.pth et G_ep200.pth>
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

from data import mnist_raw, mnist_split
from evaluate import Classifier, coverage, fid, generate, knn_radii, load_classifier, predict
from model import Generator
from utils import load_model

DEVICE = torch.device('cpu')


def train_with_seed(seed, epochs=3):
    """Même recette que evaluate.train_classifier, avec une autre seed et sans rien sauvegarder."""
    torch.manual_seed(seed)
    x, y = mnist_raw(train=True)
    x = x.unsqueeze(1).float() / 255
    clf = Classifier()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return clf.eval()


def accuracy(clf, x, y):
    """Part d'images bien classées."""
    return (predict(clf, x, DEVICE)[1] == y).float().mean().item()


@torch.no_grad()
def top_confidence(clf, x):
    """Probabilité de la classe prédite pour chaque image."""
    return torch.cat([F.softmax(clf(x[i:i + 1000]), 1).max(1).values for i in range(0, len(x), 1000)])


def scores(clf, images, real):
    """FID, précision et rappel de images contre real, dans les features de clf (k = 3)."""
    f_real, f_fake = predict(clf, real, DEVICE)[0], predict(clf, images, DEVICE)[0]
    return (fid(f_real, f_fake), coverage(f_fake, f_real, knn_radii(f_real, 3)), coverage(f_real, f_fake, knn_radii(f_fake, 3)))


def main(weights):
    x_test, y_test = mnist_split('test')
    x_val, y_val = mnist_split('val')
    x_train = mnist_split('train')[0][:10_000]
    images = {e: generate(load_model(Generator(784), os.path.join(weights, f'G_ep{e}.pth'), 'cpu').eval(), 10_000, DEVICE)
              for e in (120, 200)}

    shared = load_classifier(DEVICE)
    preds = predict(shared, x_test, DEVICE)[1]
    print('CNN du groupe : precision test', round(accuracy(shared, x_test, y_test), 4), 'val', round(accuracy(shared, x_val, y_val), 4))
    print('precision test par chiffre', [round((preds[y_test == d] == d).float().mean().item(), 3) for d in range(10)])
    for name, x in (('vraies images test', x_test), ('epoque 120', images[120]), ('epoque 200', images[200])):
        conf = top_confidence(shared, x)
        print(f'{name}: confiance moyenne {conf.mean():.3f}, part sous 0,9 : {100 * (conf < 0.9).float().mean():.1f} %')

    classifiers = {'groupe': shared}
    for seed in (1, 2):
        classifiers[f'seed {seed}'] = train_with_seed(seed)
    rows = {}
    for name, clf in classifiers.items():
        print(name, 'precision test', round(accuracy(clf, x_test, y_test), 4))
        rows[name] = {e: scores(clf, images[e], x_val) for e in images}
        rows[name]['plafond'] = scores(clf, x_train, x_val)
        rows[name]['plafond_test'] = scores(clf, x_train, x_test)
        for key, (f, p, r) in rows[name].items():
            print(f'   {key}: FID {f:.1f} P {p:.3f} R {r:.3f}')
    shares = {e: predict(clf, images[e], DEVICE)[1].bincount(minlength=10).numpy() / 100 for e in images}
    print('parts epoque 200, dernier CNN', np.round(shares[200], 1).tolist())
    print('parts epoque 200, CNN du groupe', np.round(predict(shared, images[200], DEVICE)[1].bincount(minlength=10).numpy() / 100, 1).tolist())


if __name__ == '__main__':
    main(sys.argv[1])
