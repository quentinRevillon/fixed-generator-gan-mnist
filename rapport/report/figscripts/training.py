"""Figure de la partie Training : FID Inception par chiffre (haut) et part de chaque chiffre (bas).

Usage, depuis rapport/report/ :
    python figscripts/training.py      # -> figures/training_perdigit.pdf (+ .png)
Sources : enquete/results/juliet/kid_perclass_all.csv (FID par chiffre, 1 000 images par chiffre, 3 seeds ;
          'Baseline' = baseline BCE du protocole commun, Mac),
          enquete/results/juliet/juliet2_test.csv (parts : BCE brut et BCE + EMA des mêmes runs A100, RpGAN + R1 brut).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, 'figscripts')
from style import ACCENT, MUTED, apply, save  # noqa: E402

JULIET = os.path.join('..', 'enquete', 'results', 'juliet')
DIGITS = [str(d) for d in range(10)]


def perdigit_fid(name):
    """Moyenne et écart-type sur les seeds du FID Inception de chaque chiffre pour le modèle `name`."""
    k = pd.read_csv(os.path.join(JULIET, 'kid_perclass_all.csv'))
    d = k[(k['name'] == name) & k['digit'].isin(DIGITS)]
    g = d.groupby('digit')['fid_inception']
    return g.mean().loc[DIGITS].values, g.std().loc[DIGITS].values


def digit_shares(csv, name):
    """Part moyenne (en %) de chaque chiffre parmi les 10 000 images, moyennée sur les seeds du modèle `name`."""
    df = pd.read_csv(csv)
    counts = np.array([[int(c) for c in s.split('-')] for s in df[df['name'] == name]['classes']])
    return (counts / counts.sum(axis=1, keepdims=True) * 100).mean(axis=0)


def plot_fid(ax):
    """Panneau haut : FID Inception par chiffre, BCE contre RpGAN + R1 sans et avec EMA."""
    x = np.arange(10)
    series = [('Baseline', 'baseline', MUTED, 'o', '-'),
              ('RpGAN + R1 (sans EMA)', 'new loss', ACCENT, 'o', '--'),
              ('RpGAN + R1 + EMA', 'new loss + EMA', ACCENT, 's', '-')]
    for name, label, color, marker, ls in series:
        m, s = perdigit_fid(name)
        face = 'white' if ls == '--' else color
        ax.errorbar(x, m, yerr=s, color=color, ls=ls, marker=marker, ms=3, mfc=face, lw=1, capsize=1.5,
                    label=label)
    ax.set_ylabel(r'FID$_\mathrm{Inc}$ per digit')
    ax.set_xticks(x, DIGITS)
    ax.set_ylim(0, 190)
    ax.legend(loc='upper right', ncol=1, handlelength=2.2)
    ax.text(0.01, 0.97, '(a)', transform=ax.transAxes, va='top', fontweight='bold')


def plot_shares(ax):
    """Panneau bas : part de chaque chiffre ; l'EMA (appariée, mêmes runs BCE) ne corrige pas le mélange, la perte le corrige."""
    x = np.arange(10)
    w = 0.27
    bars = [(digit_shares(os.path.join(JULIET, 'juliet2_test.csv'), 'baseline_ema G brut'), 'baseline', MUTED, None),
            (digit_shares(os.path.join(JULIET, 'juliet2_test.csv'), 'baseline_ema G_EMA'), 'baseline + EMA',
             'white', '/////'),
            (digit_shares(os.path.join(JULIET, 'juliet2_test.csv'), 'cmploc_rpgan_r1_ema G brut'), 'new loss',
             ACCENT, None)]
    for i, (v, label, color, hatch) in enumerate(bars):
        ax.bar(x + (i - 1) * w, v, w, color=color, edgecolor=MUTED if hatch else color, hatch=hatch, lw=0.5,
               label=label)
    ax.axhline(10, color='black', lw=0.6, ls=':')
    ax.set_ylim(0, 40)
    ax.set_ylabel('share of samples (%)')
    ax.set_xlabel('digit')
    ax.set_xticks(x, DIGITS)
    ax.legend(loc='upper right')
    ax.text(0.01, 0.97, '(b)', transform=ax.transAxes, va='top', fontweight='bold')


def main():
    """Trace les deux panneaux l'un sous l'autre sur une colonne et enregistre training_perdigit."""
    apply()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.3, 3.2), sharex=False)
    plot_fid(a)
    plot_shares(b)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'training_perdigit')


if __name__ == '__main__':
    main()
