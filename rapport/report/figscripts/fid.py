"""Figure de la partie « fid » : FID brut de chaque défaut, dans les deux mesures, face au score de la baseline.

Usage, depuis rapport/report/ :
    python figscripts/fid.py      # -> figures/fid_sensitivity.pdf (+ .png)

Données : ../enquete/results/juliet/sensitivity_10k.csv (Juliet, N = 10 000 images, référence MNIST test).
Colonnes utilisées : fid_inception, fid_cnn ; la ligne (gan, baseline) donne le trait vertical « baseline ».
Deux panneaux côte à côte (une échelle par FID) : une barre qui atteint le trait est aussi mauvaise que la baseline.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import ACCENT, DARK, INK, MUTED, NOTE, apply, save  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'enquete', 'results', 'juliet',
                   'sensitivity_10k.csv')
# Lignes de haut en bas : str = titre de groupe ; tuple = (experiment, condition, level, label affiché).
ROWS = ['Real images, unbalanced',
        ('classes', 'chiffres retires', '1', 'one digit missing'),
        ('classes', 'chiffres retires', '2', 'two digits missing'),
        ('classes', 'desequilibre', 'parts de la baseline', "baseline's balance"),
        'Real images, 25% damaged',
        ('pixels', 'noise0.3', '0.25', 'pixel noise'),
        ('pixels', 'blur1.0', '0.25', 'blur'),
        'Generated images',
        ('gan', 'baseline remise en proportion', '', 'baseline, rebalanced'),
        ('gan', 'rpgan_r1_ema', '', 'final model')]
BASELINE = ('gan', 'baseline', '')


def lookup(df, key, column):
    """Valeur de `column` pour la ligne (experiment, condition, level) = key ; plante si elle n'est pas unique."""
    exp, cond, level = key
    hit = df[(df.experiment == exp) & (df.condition == cond) & (df.level == level)]
    assert len(hit) == 1, key
    return hit[column].item()


def load(path):
    """Renvoie (labels, FID Inception, FID CNN, baseline Inception, baseline CNN) ; titre de groupe : valeurs NaN."""
    df = pd.read_csv(path, dtype={'level': str}).fillna({'level': ''})
    labels, inc, cnn = [], [], []
    for row in ROWS:
        group = isinstance(row, str)
        labels.append(row if group else row[3])
        inc.append(np.nan if group else lookup(df, row[:3], 'fid_inception'))
        cnn.append(np.nan if group else lookup(df, row[:3], 'fid_cnn'))
    return (labels, np.array(inc), np.array(cnn),
            lookup(df, BASELINE, 'fid_inception'), lookup(df, BASELINE, 'fid_cnn'))


def panel(ax, y, values, baseline, color, title, xmax):
    """Un panneau : barres horizontales, valeur au bout, trait vertical au score de la baseline."""
    ok = ~np.isnan(values)
    ax.barh(y[ok], values[ok], 0.62, color=color)
    for yi, v in zip(y[ok], values[ok]):
        ax.text(v + xmax * 0.02, yi, f'{v:.1f}' if v < 10 else f'{v:.0f}', va='center', fontsize=NOTE, color=DARK)
    ax.axvline(baseline, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(0, xmax)
    ax.set_title(f'{title}\nbaseline = {baseline:.0f}', fontsize=8.5, pad=4)
    ax.tick_params(axis='y', length=0)


def plot(labels, inc, cnn, base_inc, base_cnn):
    """Deux panneaux partageant les lignes : FID Inception (accent) à gauche, FID CNN (gris) à droite."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.3, 2.9), sharey=True, gridspec_kw={'wspace': 0.25})
    y = np.arange(len(labels))[::-1].astype(float)
    panel(left, y, inc, base_inc, ACCENT, r'FID$_\mathrm{Inc}$', 62)
    panel(right, y, cnn, base_cnn, MUTED, r'FID$_\mathrm{CNN}$', 210)
    left.set_yticks(y, labels)
    for tick, lab in zip(left.get_yticklabels(), labels):
        if lab in [r for r in ROWS if isinstance(r, str)]:
            tick.set_fontweight('bold')
    left.set_ylim(-0.6, y[0] + 0.6)
    fig.supxlabel('FID (lower is better)', fontsize=8.5, y=-0.01)
    return fig


if __name__ == '__main__':
    apply()
    plt.rcParams['mathtext.fontset'] = 'stix'  # indices dans la police serif du texte
    save(plot(*load(CSV)), 'fid_sensitivity')
