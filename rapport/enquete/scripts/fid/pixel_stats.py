"""Statistiques de pixels : les GAN ont-ils un fond « sale » (pixels gris) que les vraies images n'ont pas ?

Teste l'hypothèse : Inception, très sensible au bruit, verrait surtout ce défaut ; le petit CNN beaucoup moins.
Lancer depuis le worktree :
    DATA=<scratchpad>/data python <scratchpad>/scripts/fid/pixel_stats.py <scratchpad>
Sortie : <scratchpad>/results/fid/pixel_stats.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.getcwd())

import torch

from data import mnist_split
from evaluate import generate
from model import Generator
from utils import load_model

torch.set_num_threads(2)
SCRATCH = sys.argv[1]
MODELS = os.path.join(SCRATCH, 'models', 'fid')
OUT = os.path.join(SCRATCH, 'results', 'fid', 'pixel_stats.csv')


def stats(x):
    """Part de pixels gris (0,05 < v < 0,5), quasi noirs (v <= 0,05), quasi blancs (v >= 0,9), intensité moyenne."""
    return dict(grey=round(((x > 0.05) & (x < 0.5)).float().mean().item(), 4),
                black=round((x <= 0.05).float().mean().item(), 4),
                white=round((x >= 0.9).float().mean().item(), 4), mean=round(x.mean().item(), 4))


if __name__ == '__main__':
    sets = {'reel (test)': mnist_split('test')[0]}
    for name in ('cmploc_baseline_s0', 'cmploc_rpgan_r1_ema_s0'):
        G = load_model(Generator(784), os.path.join(MODELS, f'{name}_G.pth'), 'cpu').eval()
        sets[name] = generate(G, 10_000, torch.device('cpu'), seed=0)
    rows = [dict(set=name, **stats(x)) for name, x in sets.items()]
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
