"""Tableaux Markdown des diagnostics DRS (lit results/drs/*.csv).

    python tables.py
"""
import csv
import glob
import os
from collections import defaultdict

OUT = r'C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad\results\drs'
ORDER = ['bl200_prof', 'bl_s1', 'rp_s1_ema', 'rp_s0_ema', 'rp_s1_raw']
VARIANTS = ['plain', 'drs_p50', 'drs_p80', 'drs_p95', 'drs_p80_platt', 'drs_p80_featlr', 'obrs_K3', 'obrs_K3_featlr',
            'drs_p80_plain_mix', 'drs_p80_test_mix', 'plain_test_mix']


def rows(pattern):
    """Toutes les lignes des CSV qui correspondent au motif."""
    return [r for p in sorted(glob.glob(os.path.join(OUT, pattern))) for r in csv.DictReader(open(p))]


def scores_table():
    """Statistiques des notes de D, une ligne par modèle."""
    print('| Modèle | d réels (moy. ± é.-t.) | d G (moy. ± é.-t.) | AUC D | AUC D recalibré (fc3) | log E_G[e^d] | Platt a |')
    print('|---|---|---|---|---|---|---|')
    by = {r['model']: r for r in rows('scores_*.csv')}
    for m in ORDER:
        if m in by:
            r = {k: float(v) if k != 'model' else v for k, v in by[m].items()}
            print(f"| {m} | {r['d_real_mean']:.2f} ± {r['d_real_std']:.2f} | {r['d_fake_mean']:.2f} ± {r['d_fake_std']:.2f} | "
                  f"{r['auc_raw']:.3f} | {r['auc_feat']:.3f} | {r['logmeanexp_d_on_G']:.2f} | {r['platt_a']:.2f} |")


def metrics_table():
    """FID CNN (précision, rappel, acceptation) par modèle et variante ; un tirage = une valeur, séparées par /."""
    agg = defaultdict(list)
    for r in rows('metrics_cnn_*.csv'):
        agg[(r['model'], r['variant'])].append(r)
    models = [m for m in ORDER if any(k[0] == m for k in agg)]
    print('| Variante | ' + ' | '.join(models) + ' |')
    print('|---|' + '---|' * len(models))
    for v in VARIANTS:
        cells = []
        for m in models:
            g = agg.get((m, v), [])
            if not g:
                cells.append('—')
                continue
            fid = ' / '.join(f"{float(r['fid_cnn']):.1f}" for r in g)
            acc = float(g[0]['acceptance'])
            p = sum(float(r['precision_cnn']) for r in g) / len(g)
            rc = sum(float(r['recall_cnn']) for r in g) / len(g)
            tv = sum(float(r['tv_to_test']) for r in g) / len(g)
            acc_s = '' if acc != acc else f', {100 * acc:.0f} %'
            cells.append(f'{fid} (P {p:.3f}, R {rc:.3f}, TV {tv:.3f}{acc_s})')
        print(f'| {v} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    scores_table()
    print()
    metrics_table()
