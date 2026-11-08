"""Distance en variation totale entre la part des chiffres générés et MNIST test, par ligne de comparison_local.csv.

Utilisation : python classes_tv.py <comparison_local.csv>
"""
import csv
import sys
from collections import defaultdict

import numpy as np

TEST = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009], float)  # effectifs MNIST test
TEST /= TEST.sum()
groups = defaultdict(list)
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f):
        c = np.array(r['classes'].split('-'), float)
        groups[r['name']].append(0.5 * np.abs(c / c.sum() - TEST).sum())
for name, v in groups.items():
    print(f'{name:40s} TV = {np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}  per seed {np.round(v, 3)}')
