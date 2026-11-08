"""Résume ema_ablation.csv : moyenne ± écart-type (ddof=1) par config et par poids (G.pth = EMA, G_raw.pth = brut).

Utilisation :
    python summarize_ablation.py <ema_ablation.csv>
Les runs effondrés (FID > 1000) sont exclus des moyennes et comptés à part. Écart-type des proportions de chiffres en points.
"""
import csv
import re
import sys
from collections import defaultdict

import numpy as np

COLS = ('run', 'weights', 'fid', 'precision', 'recall', 'classes')


def spread(classes):
    """(min %, max %, écart-type en points) des proportions de chiffres."""
    c = np.array(list(map(int, classes.split('-'))), float)
    p = 100 * c / c.sum()
    return p.min(), p.max(), p.std()


if __name__ == '__main__':
    groups = defaultdict(list)
    for r in csv.DictReader(open(sys.argv[1]), fieldnames=COLS):
        print(f"{r['run']:34s} {r['weights']:10s} FID={float(r['fid']):8.2f} P={float(r['precision']):.3f} "
              f"R={float(r['recall']):.3f} min/max/sd={'/'.join(f'{v:.1f}' for v in spread(r['classes']))}")
        groups[(re.sub(r'_s\d$', '', r['run']), r['weights'])].append(r)
    print()
    for (cfg, w), rs in sorted(groups.items()):
        ok = [r for r in rs if float(r['fid']) < 1000]
        vals = {k: np.array([float(r[k]) for r in ok]) for k in ('fid', 'precision', 'recall')}
        sd = np.array([spread(r['classes'])[2] for r in ok])
        txt = ' '.join(f'{k}={v.mean():.2f}±{v.std(ddof=1):.2f}' for k, v in vals.items())
        print(f'{cfg:30s} {w:10s} n_ok={len(ok)}/{len(rs)} {txt} sd_classes={sd.mean():.2f}')
