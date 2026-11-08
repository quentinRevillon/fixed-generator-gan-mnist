"""Figure clé : à quoi réagit chaque FID ? Mêmes vraies images dégradées, mesurées par Inception et par le petit CNN.

Lit <scratchpad>/results/fid/audit.csv (analyze.py), écrit <scratchpad>/results/fid/fid_sensitivity.png.
    python make_figure.py <scratchpad>
Axe : part de l'excès de FID de la baseline GAN, (FID − plancher) / (FID baseline − plancher), même mesure, même N = 500.
100 % = le défaut pèse autant que tout ce qui sépare la baseline de vraies images, pour ce FID.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH = sys.argv[1]
SRC = os.path.join(SCRATCH, 'results', 'fid', 'audit.csv')
OUT = os.path.join(SCRATCH, 'results', 'fid', 'fid_sensitivity.png')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'inception': '#2a78d6', 'cnn_500': '#eb6834'}
LABELS = {'inception': 'FID Inception', 'cnn_500': 'FID petit CNN'}
# (experiment, condition, level) -> libellé ; ordre de haut en bas
CONDITIONS = [
    (('pixels', 'noise0.3', '0.25'), 'Bruit σ = 0,3 sur 25 % des images'),
    (('pixels', 'blur1.0', '0.25'), 'Flou σ = 1 sur 25 % des images'),
    (('pixels', 'erase10', '0.25'), 'Trait effacé (carré 10 px) sur 25 %'),
    (('pixels', 'blend', '0.25'), 'Chiffres « chimères » sur 25 %'),
    (('classes', 'chiffres retires', '1'), '1 chiffre absent sur 10'),
    (('classes', 'chiffres retires', '2'), '2 chiffres absents sur 10'),
    (('classes', 'desequilibre', 'parts de la baseline'), 'Proportions de la baseline (25 % de « 1 »)'),
    (('gan', 'rpgan_r1_ema', ''), 'GAN : RpGAN + R1 + EMA'),
    (('gan', 'baseline remise en proportion', ''), 'GAN : baseline, chiffres remis en proportion'),
]


def load_ratios():
    """{(setting, experiment, condition, level): part de l'excès de la baseline, en %} (plancher à 0,3 % pour le log)."""
    rows = csv.DictReader(open(SRC, encoding='utf8'))
    return {(r['setting'], r['experiment'], r['condition'], r['level']): max(100 * float(r['part_baseline']), 0.3)
            for r in rows if r.get('part_baseline')}


if __name__ == '__main__':
    ratios = load_ratios()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ys = list(range(len(CONDITIONS)))[::-1]
    for y, (key, label) in zip(ys, CONDITIONS):
        vals = {s: ratios[(s, *key)] for s in COLORS}
        ax.plot([vals['inception'], vals['cnn_500']], [y, y], color=GRID, lw=2, zorder=1)
        for s, v in vals.items():
            ax.scatter(v, y, s=70, color=COLORS[s], edgecolor='white', linewidth=2, zorder=3,
                       label=LABELS[s] if y == ys[0] else None)
        for s, v in vals.items():
            right = v == max(vals.values())
            ax.annotate(f'{v:.0f} %' if v >= 1 else f'{v:.1f} %'.replace('.', ','), (v, y), xytext=(9 if right else -9, 0),
                        textcoords='offset points', va='center', ha='left' if right else 'right', fontsize=9, color=MUTED)
    ax.set_yticks(ys, [label for _, label in CONDITIONS], fontsize=10, color=INK)
    ax.set_xscale('log')
    ax.axvline(100, color=MUTED, lw=1, ls=':')
    ax.set_xlim(0.2, 600)
    ax.set_xticks([1, 10, 100], ['1 %', '10 %', '100 % = défaut total de la baseline'])
    ax.set_xlabel("Poids du défaut, en % de l'écart baseline GAN – vraies images (même FID, mêmes 500 images, échelle log)",
                  color=MUTED, fontsize=9)
    ax.grid(axis='x', color=GRID, lw=0.8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.07), ncol=2, fontsize=10)
    fig.suptitle('Mêmes images, deux FID : le CNN voit les chiffres, Inception voit le bruit',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print('ecrit', OUT)
