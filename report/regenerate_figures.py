"""Rebuild the report assets from the project repository and check every number of the tables.

Run from anywhere:  python3 regenerate_figures.py [path/to/masteriasd-datalab-project-a2-group-3]

1. Figures: copied from the group's figure scripts output (`rapport/report/figures/` on the branch
   `quentin/comparaison`, drawn by `rapport/report/figscripts/*.py`).
2. Data: the result CSVs used by the tables are copied into `data/`.
3. Check: the means, standard deviations, collapse counts and p-values quoted in `tabs/` and `sec_merge/`
   are recomputed from `data/` and printed, so they can be compared with the .tex files.
"""
import csv
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent   # the repository root (report/ lives in it)
BRANCH = 'origin/quentin/comparaison'
FIGURES = ['combination_samples', 'fid_sensitivity', 'training_perdigit']
JULIET = ['juliet2_test', 'drs_recalib_test', 'nico_data_test', 'noise_rerun_test', 'kid_perclass_all', 'sensitivity_10k']


def git_show(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{BRANCH}:{path}'], check=True, capture_output=True).stdout


def fetch_assets():
    (HERE / 'figs').mkdir(exist_ok=True)
    (HERE / 'data').mkdir(exist_ok=True)
    for name in FIGURES:
        (HERE / 'figs' / f'{name}.pdf').write_bytes(git_show(f'rapport/report/figures/{name}.pdf'))
    for name in JULIET:
        (HERE / 'data' / f'{name}.csv').write_bytes(git_show(f'rapport/enquete/results/juliet/{name}.csv'))
    (HERE / 'data' / 'ema_summary.csv').write_bytes(git_show('rapport/enquete/results/ema/ema_summary.csv'))
    (HERE / 'data' / 'comparison_local.csv').write_bytes(git_show('results/comparison_local.csv'))


def rows(name):
    with open(HERE / 'data' / f'{name}.csv', newline='') as f:
        return list(csv.DictReader(f))


def summary(group, label):
    fid = [float(r['fid']) for r in group]
    sd = st.stdev(fid) if len(fid) > 1 else 0.0
    extra = ' '.join(f'{k}={st.mean(float(r[k]) for r in group):.2f}'
                     for k in ('precision', 'recall', 'fid_cnn', 'acceptance_rate') if k in group[0])
    print(f'  {label:44s} n={len(group):2d}  FID_Inc {st.mean(fid):6.2f} +- {sd:4.2f}  {extra}')


def by(table, key):
    groups = defaultdict(list)
    for r in table:
        groups[key(r)].append(r)
    return groups


def check():
    healthy = lambda r: float(r['recall_cnn']) >= 0.01   # collapsed run: classifier recall < 0.01
    print('Table 1 (data lever), selection, non-collapsed runs:')
    for name, group in by(rows('nico_data_test'), lambda r: r['name']).items():
        ok = [r for r in group if healthy(r)]
        if name in ('data_ref', 'sel_95', 'sel_90', 'sel_75', 'sel_50') and ok:
            summary(ok, f'{name} (collapsed {len(group) - len(ok)}/{len(group)})')
    print('Tables 1-3, common protocol (A100 runs):')
    for name, group in by(rows('juliet2_test'), lambda r: r['name']).items():
        ok = [r for r in group if healthy(r)]
        summary(ok, f'{name} (collapsed {len(group) - len(ok)}/{len(group)})')
    print('Tables 1-3, common protocol (M3 runs):')
    for name, group in by(rows('comparison_local'), lambda r: r['name']).items():
        summary(group, name)
    print('Table 4 (rejection), 3 seeds x 2 draws:')
    for (model, variant), group in by(rows('drs_recalib_test'), lambda r: (re.sub(r'_s\d$', '', r['model']), r['variant'])).items():
        summary(group, f'{model} / {variant} (AUC {st.mean(float(r["auc"]) for r in group):.2f})')
    try:
        from scipy import stats
    except ImportError:
        print('scipy not installed: p-values skipped')
        return
    seed = lambda r: re.search(r'_s(\d)', r['run']).group(1)
    pick = lambda table, name: {seed(r): float(r['fid']) for r in table if r['name'] == name}
    J, C = rows('juliet2_test'), rows('comparison_local')
    bce_ema, bce_raw = pick(J, 'baseline_ema G_EMA'), pick(J, 'baseline_ema G brut')
    rp_ema, rp_raw = pick(C, 'RpGAN + R1 + EMA'), pick(J, 'cmploc_rpgan_r1_ema G brut')
    s = sorted(bce_ema)
    print('p-values:')
    print('  EMA on baseline loss (paired)', stats.ttest_rel([bce_ema[k] for k in s], [bce_raw[k] for k in s]).pvalue)
    print('  EMA on RpGAN + R1 (paired)   ', stats.ttest_rel([rp_ema[k] for k in s], [rp_raw[k] for k in s]).pvalue)
    print('  EMA gain, loss vs baseline   ', stats.ttest_ind([bce_ema[k] - bce_raw[k] for k in s],
                                                             [rp_ema[k] - rp_raw[k] for k in s], equal_var=False).pvalue)
    print('  loss effect with EMA (Welch) ', stats.ttest_ind(list(bce_ema.values()), list(rp_ema.values()), equal_var=False).pvalue)
    noise = [float(r['fid']) for r in J if r['name'] == 'baseline_noise_03_ema G_EMA' and healthy(r)]
    print('  noise, baseline loss (Welch) ', stats.ttest_ind(noise, list(bce_ema.values()), equal_var=False).pvalue)
    print('  noise, RpGAN + R1 (Welch)    ', stats.ttest_ind(list(pick(C, 'RpGAN + R1 + EMA + bruit 0,3').values()),
                                                             list(rp_ema.values()), equal_var=False).pvalue)
    R = rows('drs_recalib_test')
    per_seed = lambda v: [st.mean(float(r['fid']) for r in R if r['model'] == m and r['variant'] == v)
                          for m in ('rpgan_r1_ema_s0', 'rpgan_r1_ema_s1', 'rpgan_r1_ema_s2')]
    print('  recalibrated OBRS vs none    ', stats.ttest_rel(per_seed('obrs_K3_recal'), per_seed('plain')).pvalue)


if __name__ == '__main__':
    if (REPO / '.git').exists():
        fetch_assets()
    else:
        print(f'{REPO} not found: using the files already in figs/ and data/')
    check()
