"""Figure de la partie « fid » : poids de chaque défaut dans les deux FID (part de l'écart baseline - vraies images).

Usage, depuis rapport/report/ :
    python figscripts/fid.py      # -> figures/fid_sensitivity.pdf (+ .png)

Données : ../enquete/results/juliet/sensitivity_10k.csv (Juliet, N = 10 000 images, référence MNIST test).
Colonne utilisée : *_part_baseline = (FID - plancher) / (FID baseline GAN - plancher), dans la même mesure.
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
        ('classes', 'desequilibre', 'parts de la baseline', "baseline's digit balance"),
        'Real images, 25% damaged',
        ('pixels', 'noise0.3', '0.25', 'pixel noise'),
        ('pixels', 'blur1.0', '0.25', 'blur'),
        'Generated images',
        ('gan', 'baseline remise en proportion', '', 'baseline, rebalanced'),
        ('gan', 'rpgan_r1_ema', '', 'final model')]


def load_shares(path):
    """Renvoie (labels, part Inception en %, part CNN en %) ; pour un titre de groupe, label = titre et parts NaN."""
    df = pd.read_csv(path, dtype={'level': str}).fillna({'level': ''})
    labels, inc, cnn = [], [], []
    for row in ROWS:
        if isinstance(row, str):
            labels.append(row), inc.append(np.nan), cnn.append(np.nan)
            continue
        exp, cond, level, label = row
        hit = df[(df.experiment == exp) & (df.condition == cond) & (df.level == level)]
        assert len(hit) == 1, row
        labels.append(label)
        inc.append(100 * hit.fid_inception_part_baseline.item())
        cnn.append(100 * hit.fid_cnn_part_baseline.item())
    return labels, np.array(inc), np.array(cnn)


def plot(labels, inc, cnn):
    """Barres horizontales groupées : FID_Inc (accent) et FID_CNN (gris), valeur au bout de chaque barre en gris foncé,
    titres de groupe dans le graphique, légende horizontale au-dessus des axes, ligne 100 % étiquetée."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    y = np.arange(len(labels))[::-1].astype(float)
    h = 0.42
    ax.barh(y + h / 2, inc, h, color=ACCENT, label=r'FID$_\mathrm{Inc}$')
    ax.barh(y - h / 2, cnn, h, color=MUTED, label=r'FID$_\mathrm{CNN}$')
    ticks, names = [], []
    for yi, lab, a, b in zip(y, labels, inc, cnn):
        if np.isnan(a):
            ax.text(0.01, yi - 0.05, lab, transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=8.5, fontweight='bold', color=INK)
            continue
        ticks.append(yi), names.append(lab)
        ax.text(a + 1.5, yi + h / 2, f'{a:.0f}', va='center', fontsize=NOTE, color=DARK)
        ax.text(b + 1.5, yi - h / 2, f'{b:.0f}', va='center', fontsize=NOTE, color=DARK)
    ax.set_yticks(ticks, names)
    ax.axvline(100, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.text(101.5, y[2], '100% =\nas bad as\nthe baseline', ha='left', va='center', fontsize=NOTE, color=DARK)
    ax.set_xlim(0, 132)
    ax.set_ylim(-0.6, y[0] + 0.5)
    ax.set_xlabel("FID above the floor, as % of the baseline's")
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, borderaxespad=0.2)
    return fig


if __name__ == '__main__':
    apply()
    plt.rcParams['mathtext.fontset'] = 'stix'  # sigma et indices dans la police serif du texte
    save(plot(*load_shares(CSV)), 'fid_sensitivity')
