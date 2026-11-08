"""Features Inception (pytorch-fid, 2048 dim) mises en cache en .npy : MNIST test complet, puis des générateurs.

Utilisation (depuis n'importe où ; DATA doit pointer vers le dossier torchvision de MNIST) :
    DATA=<data> python inception_feats.py <code_dir> <cache_dir> <n> real run/G_raw.pth run/G.pth ...
    'real'          -> <cache_dir>/real_test.npy        (les 10 000 images de mnist_split('test'), comme le groupe)
    '<run>/<poids>' -> <cache_dir>/<run>__<poids>__n<n>.npy  (n images de evaluate.generate, z de la seed 0)
Un fichier déjà présent est sauté : on peut relancer sans refaire le travail.
La génération est celle de scripts/measure_runs.py sans DRS (sample 'plain', seed 0 == evaluate.generate, seed 0).
"""
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.getenv('N_THREADS', 4)))
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from data import mnist_split  # noqa: E402
from evaluate import generate, inception_features  # noqa: E402
from model import Generator  # noqa: E402
from utils import load_model  # noqa: E402

DEVICE = torch.device('cpu')


def generator_images(spec, n):
    """n images (n, 1, 28, 28) dans [0, 1] du G stocké dans runs/<spec>."""
    G = load_model(Generator(784).to(DEVICE), os.path.join('runs', spec), DEVICE)
    return generate(G, n, DEVICE)


def cache_path(cache_dir, spec, n):
    """Chemin du .npy d'une entrée ('real' ou '<run>/<poids>')."""
    if spec == 'real':
        return os.path.join(cache_dir, 'real_test.npy')
    return os.path.join(cache_dir, f"{spec.replace('/', '__')}__n{n}.npy")


def compute_features(spec, n):
    """Features Inception (N, 2048) de l'entrée spec."""
    images = mnist_split('test')[0] if spec == 'real' else generator_images(spec, n)
    return inception_features(images, DEVICE).numpy()


if __name__ == '__main__':
    cache_dir, n, specs = sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    os.makedirs(cache_dir, exist_ok=True)
    for spec in specs:
        path = cache_path(cache_dir, spec, n)
        if os.path.exists(path):
            print(f'{spec} : déjà en cache', flush=True)
            continue
        t0 = time.time()
        np.save(path, compute_features(spec, n))
        print(f'{spec} : {time.time() - t0:.0f}s -> {path}', flush=True)
