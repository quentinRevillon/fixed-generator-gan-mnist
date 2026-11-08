"""Statistiques du volet Entraînement à partir des logs de train.py (branche noe/training, extraite en lecture).

Utilisation :
    python log_stats.py <dossier_logs> <results.csv> <comparison_local.csv>

Affiche : (1) FID/P/R finaux par run et moyenne ± écart-type (ddof=1) par config ; (2) courbes FID toutes les
10 époques ; (3) premières époques où D sature (D(x) >= 0.99 et D(G(z)) <= 0.01) ; (4) stats du protocole commun.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

LINE = re.compile(r'^\[(\d+)/(\d+)\] D=([\d.]+) G=([\d.]+) D\(x\)=([\d.]+) D\(G\(z\)\)=([\d.]+)'
                  r'(?: \| FID=([\d.]+) P=([\d.]+) R=([\d.]+))? \| ([\d.]+)s/ep')


def parse_log(path):
    """Liste de dicts par époque : epoch, loss_D, loss_G, D_real, D_fake, fid, P, R, sec."""
    rows = []
    for line in open(path, encoding='utf-8', errors='replace'):
        m = LINE.match(line)
        if m:
            g = m.groups()
            rows.append(dict(epoch=int(g[0]), loss_D=float(g[2]), loss_G=float(g[3]), D_real=float(g[4]),
                             D_fake=float(g[5]), fid=float(g[6]) if g[6] else None,
                             P=float(g[7]) if g[7] else None, R=float(g[8]) if g[8] else None, sec=float(g[9])))
    return rows


def last_by_epoch(rows):
    """Garde la dernière occurrence de chaque époque (les reprises --resume réécrivent des époques)."""
    return [r for _, r in sorted({r['epoch']: r for r in rows}.items())]


def mean_sd(values):
    """'moyenne ± écart-type (ddof=1)'."""
    v = np.array(values, float)
    return f'{v.mean():.2f} ± {v.std(ddof=1):.2f}' if len(v) > 1 else f'{v.mean():.2f}'


def config_of(run):
    """Nom de config sans le suffixe _s<seed>."""
    return re.sub(r'_s\d$', '', run)


def report_logs(log_dir):
    """Résumé et courbes de tous les logs ema100_* et cmploc_*."""
    finals = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, 'ema100_*.log')) + glob.glob(os.path.join(log_dir, 'cmploc_*.log'))
                       + glob.glob(os.path.join(log_dir, 'train_*.log')) + glob.glob(os.path.join(log_dir, 'baseline_rerun.log'))):
        run = os.path.basename(path)[:-4]
        rows = last_by_epoch(parse_log(path))
        ev = [r for r in rows if r['fid'] is not None]
        curve = ' '.join(f"{r['epoch']}:{r['fid']:.1f}" for r in ev)
        sat = next((r['epoch'] for r in rows if r['D_real'] >= 0.99 and r['D_fake'] <= 0.01), None)
        sec = np.median([r['sec'] for r in rows])
        best = min(ev, key=lambda r: r['fid'])
        print(f"{run:34s} last_ep={rows[-1]['epoch']:3d} FID={ev[-1]['fid']:.2f} P={ev[-1]['P']:.3f} R={ev[-1]['R']:.3f} "
              f"best={best['fid']:.2f}@{best['epoch']} sat_ep={sat} med={sec:.1f}s/ep "
              f"lastD(x)={rows[-1]['D_real']:.2f} lastD(Gz)={rows[-1]['D_fake']:.2f}")
        print(f"    FID : {curve}")
        finals[config_of(run)].append((ev[-1]['fid'], ev[-1]['P'], ev[-1]['R'], rows[-1]['epoch']))
    print('\n== Moyennes par config (dernière évaluation) ==')
    for cfg, vals in finals.items():
        fids, ps, rs, eps = zip(*vals)
        ok = [v for v in vals if v[0] < 1000]
        line = f'{cfg:30s} n={len(vals)} FID={mean_sd(fids)} P={mean_sd(ps)} R={mean_sd(rs)} epochs={eps}'
        if ok and len(ok) < len(vals):
            f2, p2, r2, _ = zip(*ok)
            line += f' | sans effondrés (n={len(ok)}) FID={mean_sd(f2)} P={mean_sd(p2)} R={mean_sd(r2)}'
        print(line)


def report_common(path):
    """Moyenne ± écart-type par méthode du protocole commun (results/comparison_local.csv)."""
    groups = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        groups[r['name']].append(r)
    print('\n== Protocole commun (test) ==')
    for name, rs in groups.items():
        cols = {k: [float(r[k]) for r in rs] for k in ('fid', 'precision', 'recall', 'fid_cnn', 'precision_cnn', 'recall_cnn')}
        print(f"{name:38s} n={len(rs)} " + ' '.join(f'{k}={mean_sd(v)}' for k, v in cols.items()))


def class_spread(path):
    """Proportions de classes (min, max, écart-type en points) par run de results.csv."""
    print('\n== Proportions de chiffres (results.csv) ==')
    for r in csv.DictReader(open(path, encoding='utf-8')):
        c = np.array(list(map(int, r['classes'].split('-'))), float)
        p = 100 * c / c.sum()
        print(f"{r['run']:28s} FID={float(r['fid']):8.2f} min={p.min():5.1f}% (chiffre {p.argmin()}) "
              f"max={p.max():5.1f}% (chiffre {p.argmax()}) sd={p.std():.2f} pts")


if __name__ == '__main__':
    report_logs(sys.argv[1])
    class_spread(sys.argv[2])
    report_common(sys.argv[3])
