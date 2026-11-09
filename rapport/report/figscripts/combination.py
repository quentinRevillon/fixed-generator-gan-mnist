"""Figure of part "combination": gain in FID_Inc of each add-on, depending on the model it is added to.

Run from rapport/report/:
    python figscripts/combination.py      # -> figures/combination_effects.pdf (+ .png)

Each row (all printed; those in NOT_DRAWN are not drawn) is one add-on on one base model: Delta FID_Inc = FID_Inc(base + add-on) - FID_Inc(base), test set.
Paired effects (same trained G, with and without the add-on) show one dot per seed; unpaired effects
(different trained runs) show the mean only. Blue: p < 0.05 (paired t-test, or Welch for unpaired), grey: not.
Sources: results/comparison_local.csv (common protocol, Mac), enquete/results/juliet/{juliet2_test,drs_recalib_test}.csv.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, 'figscripts')
from style import ACCENT, DARK, MUTED, NOTE, apply, save  # noqa: E402

JULIET = os.path.join('..', 'enquete', 'results', 'juliet')
COMMON = os.path.join('..', '..', 'results', 'comparison_local.csv')
# Rows printed but not drawn (stated in the text and Table 1, left out to fit the page budget).
NOT_DRAWN = {('Recalibrated D', 'DRS on BCE'), ('Noise 0.3', 'on RpGAN + R1 + EMA'), ('Noise 0.3', 'on BCE + EMA (3/5 runs)')}


def by_seed(df, name):
    """FID_Inc per training seed for one configuration name of a CSV with columns name, run, fid."""
    rows = df[df.name == name]
    return rows.set_index(rows.run.str.extract(r'_s(\d)/')[0].astype(int)).fid.sort_index()


def recal_by_seed(df, base, variant):
    """FID_Inc per seed (mean over draws) for one base G and one sampler of drs_recalib_test.csv."""
    rows = df[df.model.str.startswith(base) & (df.variant == variant)]
    return rows.groupby(rows.model.str[-1].astype(int)).fid.mean()


def paired(after, before):
    """Paired effect: (mean difference, per-seed differences, p of the paired t-test)."""
    diff = after - before
    return diff.mean(), diff.values, stats.ttest_rel(after, before).pvalue


def unpaired(after, before):
    """Unpaired effect between independent runs: (mean difference, no seed dots, Welch p)."""
    return after.mean() - before.mean(), [], stats.ttest_ind(after, before, equal_var=False).pvalue


def effects():
    """List of (group, label, (mean, seed differences, p)) for every add-on measured on several bases."""
    common, j2 = pd.read_csv(COMMON), pd.read_csv(os.path.join(JULIET, 'juliet2_test.csv'))
    recal = pd.read_csv(os.path.join(JULIET, 'drs_recalib_test.csv'))
    bce, rp = by_seed(common, 'Baseline commune'), by_seed(common, 'RpGAN + R1 + EMA')
    bce_ema, bce_raw = by_seed(j2, 'baseline_ema G_EMA'), by_seed(j2, 'baseline_ema G brut')
    rp_raw = by_seed(j2, 'cmploc_rpgan_r1_ema G brut')
    rp_j = recal_by_seed(recal, 'rpgan', 'plain')
    noise_bce = by_seed(j2, 'baseline_noise_03_ema G_EMA')
    noise_bce = noise_bce[noise_bce < 100]  # healthy runs only: 2 of 5 collapsed (FID_Inc > 400)
    sampler = lambda g, v: paired(recal_by_seed(recal, g, v), recal_by_seed(recal, g, 'plain'))  # noqa: E731
    return [
        ('EMA', 'on BCE', paired(bce_ema, bce_raw)),
        ('EMA', 'on RpGAN + R1', paired(rp_j, rp_raw)),
        ('RpGAN + R1 loss', 'without EMA', unpaired(rp_raw, bce)),
        ('RpGAN + R1 loss', 'with EMA', unpaired(rp, bce_ema)),
        ('DRS, D as trained', 'on BCE', sampler('baseline', 'drs_p80')),
        ('DRS, D as trained', 'on BCE + EMA', paired(by_seed(j2, 'baseline_ema G_EMA + DRS'), bce_ema)),
        ('DRS, D as trained', 'on RpGAN + R1 + EMA', sampler('rpgan', 'drs_p80')),
        ('Recalibrated D', 'DRS on BCE', sampler('baseline', 'drs_p80_recal')),
        ('Recalibrated D', 'DRS on RpGAN + R1 + EMA', sampler('rpgan', 'drs_p80_recal')),
        ('Recalibrated D', 'OBRS on RpGAN + R1 + EMA', sampler('rpgan', 'obrs_K3_recal')),
        ('Noise 0.3', 'on RpGAN + R1 + EMA', unpaired(by_seed(common, 'RpGAN + R1 + EMA + bruit 0,3'), rp)),
        ('Noise 0.3', 'on BCE + EMA (3/5 runs)', unpaired(noise_bce, bce_ema)),
    ]


def plot(rows):
    """Forest plot: one line per (add-on, base), under a bold title naming the add-on, right-aligned with the row
    labels; half a row of extra space between groups; zero line = no effect."""
    apply()
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    y, ticks, labels = 0, [], []
    for i, (group, label, (mean, seeds, p)) in enumerate(rows):
        if not i or group != rows[i - 1][0]:
            y += 0.5 if i else 0
            ax.annotate(group.replace(' D', ' $D$'), (0, y), xycoords=('axes fraction', 'data'), xytext=(-3.5, 0),
                        textcoords='offset points', ha='right', va='center', fontsize=8.5, fontweight='bold')
            y += 1
        color = ACCENT if p < 0.05 else MUTED
        ax.scatter(seeds, [y] * len(seeds), s=7, color=color, alpha=0.5, lw=0)
        ax.scatter([mean], [y], s=20, color=color, zorder=3)
        ax.text(1.01, y, f'{mean:+.1f}'.replace('-', '−'), transform=ax.get_yaxis_transform(), va='center',
                fontsize=NOTE, color=ACCENT if p < 0.05 else DARK)
        ticks.append(y)
        labels.append(label)
        y += 1
    ax.axvline(0, color='black', lw=0.6)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(-17.5, 1.5)
    ax.set_xlabel(r'$\Delta\,\mathrm{FID}_{\mathrm{Inc}}$ from the add-on (lower = better)')
    ax.spines['left'].set_visible(False)
    return fig


if __name__ == '__main__':
    rows = effects()
    for group, label, (mean, seeds, p) in rows:
        print(f'{group:18s} {label:24s} {mean:+6.2f}  p={p:.3f}  seeds={[round(s, 2) for s in seeds]}')
    seeds = {r[:2]: r[2][1] for r in rows}  # interaction tests quoted in the text: Welch on per-seed effects
    for a, b in [(('EMA', 'on BCE'), ('EMA', 'on RpGAN + R1')),
                 (('DRS, D as trained', 'on BCE'), ('DRS, D as trained', 'on RpGAN + R1 + EMA'))]:
        print(f'interaction {a[0]}: Welch p={stats.ttest_ind(seeds[a], seeds[b], equal_var=False).pvalue:.3f}')
    save(plot([r for r in rows if r[:2] not in NOT_DRAWN]), 'combination_effects')
