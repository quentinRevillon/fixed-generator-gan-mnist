"""FID Inception (pytorch-fid, 2048 dim) de G.pth (EMA) et G_raw.pth (brut) de runs du protocole commun, contre MNIST test.

Utilisation :
    python ema_inception.py <code_dir> <out.csv> <n_images> run1 [run2 ...]
    WEIGHTS=G.pth python ema_inception.py ...   # autres poids que G_raw.pth,G.pth (ex. baseline sans EMA)
Même génération que scripts/measure_runs.py sans DRS (z de la seed 0 via evaluate.generate). Pas de VGG (trop lent sur CPU).
"""
import csv
import os
import sys
import time

import torch

torch.set_num_threads(int(os.getenv('N_THREADS', 2)))
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from data import mnist_split  # noqa: E402
from evaluate import fid, generate, inception_features  # noqa: E402
from model import Generator  # noqa: E402
from utils import load_model  # noqa: E402


def inception_fid(path, f_real, n, device):
    """FID Inception de n images du G stocké dans path."""
    G = load_model(Generator(784).to(device), path, device)
    return fid(f_real, inception_features(generate(G, n, device), device))


if __name__ == '__main__':
    device = torch.device('cpu')
    out, n, runs = sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    t0 = time.time()
    f_real = inception_features(mnist_split('test')[0][:n], device)
    print(f'features réelles : {time.time() - t0:.0f}s', flush=True)
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        for run in runs:
            for weights in os.getenv('WEIGHTS', 'G_raw.pth,G.pth').split(','):
                row = (run, weights, n, inception_fid(os.path.join('runs', run, weights), f_real, n, device))
                w.writerow(row)
                f.flush()
                print(*row, f'{time.time() - t0:.0f}s', flush=True)
