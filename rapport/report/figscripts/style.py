"""Style commun des figures du rapport : importer, appeler apply() avant de tracer, puis save().

    import sys; sys.path.insert(0, 'figscripts')
    from style import apply, save, ACCENT, ALERT, GOOD, MUTED
    apply(); fig, ax = plt.subplots(figsize=(3.3, 2.0)); ...; save(fig, 'fid_sensitivity')
Largeur d'une colonne : 3.3 pouces ; pleine largeur : 6.9 pouces. Texte 8 pt. Un seul accent pour les données.
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ACCENT, ALERT, GOOD, MUTED, INK = '#1F6FC5', '#D55E00', '#009E73', '#8A8984', '#0B0B0B'
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def apply():
    """Réglages matplotlib : police serif proche du texte, 8 pt, sans cadre haut et droit."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False,
                         'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def save(fig, name):
    """Enregistre figures/<name>.pdf (vectoriel, pour LaTeX) et figures/<name>.png (aperçu)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'))
    plt.close(fig)
