"""Ablation EMA sans réentraînement : compare G.pth (G moyenné, EMA) et G_raw.pth (G brut) d'un même run.

Avec ema.decay, train.py n'utilise jamais G_ema pendant l'entraînement (train.py:79-80 de noe/training) :
la trajectoire du G brut est donc exactement celle d'un run sans EMA. Comparer G.pth et G_raw.pth isole l'EMA.

Utilisation (depuis le dossier qui contient evaluate.py, data.py, runs/, checkpoints/) :
    python ema_ablation.py <code_dir> <out.csv> run1 run2 ...
Mesure : petit CNN (evaluate.evaluate), 10 000 images, z de la seed 0, contre MNIST test.
"""
import csv
import os
import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from evaluate import evaluate, generate  # noqa: E402
from model import Generator  # noqa: E402
from utils import load_model  # noqa: E402


def measure(path, device):
    """FID / precision / recall / classes du petit CNN pour un fichier de poids de G."""
    G = load_model(Generator(784).to(device), path, device)
    m = evaluate(generate(G, 10_000, device), device, k=3, reference='test')
    return m['fid'], m['precision'], m['recall'], '-'.join(map(str, m['classes']))


if __name__ == '__main__':
    device = torch.device('cpu')
    out, runs = sys.argv[2], sys.argv[3:]
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        for run in runs:
            for weights in ('G.pth', 'G_raw.pth'):
                row = (run, weights, *measure(os.path.join('runs', run, weights), device))
                w.writerow(row)
                f.flush()
                print(*row, flush=True)
