"""Figure of the Data part: precision vs recall of the instance-selection runs, in VGG (platform) and CNN features.

Usage, from rapport/report/:
    python figscripts/data.py        # -> figures/data_selection_pr.pdf (+ .png preview)

Source: ../enquete/results/juliet/nico_data_test.csv (Nicodeme's 52 generators, last checkpoint, against MNIST test).
A run is collapsed when its recall is below 0.01; only healthy runs are averaged.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'figscripts')
from style import apply, save, ACCENT, DARK, MUTED, NOTE  # noqa: E402

CSV = os.path.join('..', 'enquete', 'results', 'juliet', 'nico_data_test.csv')
RUNS = [('data_ref', '100%'), ('sel_95', '95%'), ('sel_90', '90%'), ('sel_75', '75%'), ('sel_50', '50%')]


def healthy_stats(df):
    """Mean and sd of P/R (VGG and CNN) over the healthy runs of each selection rate, in RUNS order."""
    healthy = df[df.recall >= 0.01]
    cols = ['precision', 'recall', 'precision_cnn', 'recall_cnn']
    stats = healthy.groupby('name')[cols].agg(['mean', 'std'])
    return stats.loc[[name for name, _ in RUNS]]


# Direct labels (no legend): (text, anchor point in data units, offset in points, ha, va, colour key).
LABELS = [('VGG (platform), 50%', (-0.199, 0.050), (-6, -1), 'right', 'top', 'vgg'),
          ('75%', (-0.122, 0.032), (4, -4), 'left', 'top', 'vgg'),
          ('CNN (selection), 50%', (-0.368, 0.081), (-3, 6), 'left', 'bottom', 'cnn'),
          ('75%', (-0.160, 0.048), (0, 6), 'center', 'bottom', 'cnn'),
          ('90, 95%', (-0.040, 0.000), (-8, -12), 'right', 'top', 'cnn'),
          ('ref.', (0.000, 0.000), (4, -6), 'left', 'top', 'cnn')]


def plot_space(ax, stats, p, r, color, marker):
    """Draws one trajectory (100% -> 50%) as changes from the reference (Delta recall, Delta precision), sd bars."""
    dr = stats[(r, 'mean')] - stats[(r, 'mean')].iloc[0]
    dp = stats[(p, 'mean')] - stats[(p, 'mean')].iloc[0]
    ax.errorbar(dr, dp, xerr=stats[(r, 'std')], yerr=stats[(p, 'std')], color=color, marker=marker, ms=4, lw=1,
                elinewidth=0.7, ecolor=color, capsize=0, mfc=color if marker == 'o' else 'white',
                zorder=3 if marker == 'o' else 2)
    for line in ax.collections[-2:]:
        line.set_alpha(0.6)


def main():
    """Builds figures/data_selection_pr.pdf; points are labelled directly (selection rate), in their series colour."""
    apply()
    stats = healthy_stats(pd.read_csv(CSV))
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    plot_space(ax, stats, 'precision', 'recall', ACCENT, 'o')
    plot_space(ax, stats, 'precision_cnn', 'recall_cnn', MUTED, 's')
    ax.axhline(0, color=MUTED, lw=0.5, ls=':')
    ax.axvline(0, color=MUTED, lw=0.5, ls=':')
    for text, xy, offset, ha, va, series in LABELS:
        ax.annotate(text, xy, xytext=offset, textcoords='offset points', fontsize=NOTE, ha=ha, va=va,
                    color=ACCENT if series == 'vgg' else DARK)
    ax.set_xlim(-0.42, 0.06)
    ax.set_ylim(-0.06, 0.13)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r'$\Delta$ recall vs reference (diversity)')
    ax.set_ylabel(r'$\Delta$ precision (fidelity)')
    save(fig, 'data_selection_pr')


if __name__ == '__main__':
    main()
