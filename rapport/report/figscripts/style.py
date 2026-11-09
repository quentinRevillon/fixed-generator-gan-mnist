"""Style commun des figures du rapport : importer, appeler apply() avant de tracer, puis save().

    import sys; sys.path.insert(0, 'figscripts')
    from style import apply, save, ACCENT, ALERT, GOOD, MUTED
    apply(); fig, ax = plt.subplots(figsize=(3.3, 2.0)); ...; save(fig, 'fid_sensitivity')  # save(fig, name, FULL) : pleine page
save() ramène la boîte serrée de la figure à 3,3 pouces (une colonne) ou 6,9 (FULL) : pas de mise à l'échelle dans LaTeX.
Texte 8,5 pt, annotations 8 pt en gris foncé (DARK). Bleu (ACCENT) réservé à « significatif » ou « notre méthode ».
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ACCENT, ALERT, GOOD, MUTED, INK = '#1F6FC5', '#D55E00', '#009E73', '#8A8984', '#0B0B0B'
DARK = '#444444'  # annotations (valeurs, étiquettes directes) : lisibles à l'impression
NOTE = 8  # taille des annotations dans le graphique (pt)
COLUMN, FULL = 3.3, 6.9  # largeur utile (pouces) d'une colonne, de la page ; + 2 x 0,02 de marge
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def apply():
    """Réglages matplotlib : police serif (texte et maths) proche du texte, 8,5 pt (annotations : NOTE = 8 pt), sans cadre haut et droit."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False,
                         'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'axes.labelsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def fit_width(fig, width):
    """Ajuste la largeur de la figure pour que sa boîte serrée (étiquettes comprises) mesure `width` pouces."""
    for _ in range(4):
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).width
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + width - tight, h)


def save(fig, name, width=COLUMN):
    """Enregistre figures/<name>.pdf (vectoriel, pour LaTeX) et figures/<name>.png (aperçu), à la largeur `width`
    (une colonne par défaut, FULL pour toute la page) : les polices gardent leur taille une fois dans LaTeX."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fit_width(fig, width)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'))
    plt.close(fig)
