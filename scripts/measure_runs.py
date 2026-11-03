"""Mesure plusieurs générateurs de la même façon : FID Inception, précision et rappel VGG, contre MNIST test.

Le manifeste est un CSV sans en-tête, une ligne par configuration : nom,chemin_de_G[,chemin_de_D]
Avec D, on mesure le DRS au percentile 80 ; sans D, les images de G sans tri.

Utilisation, depuis la racine du repo :
    PYTHONPATH=. python scripts/measure_runs.py <manifeste.csv> <sortie.csv>
"""
import csv
import sys

from evaluate import evaluate_prof, to_unit_images
from model import Discriminator, Generator
from sampling import sample
from utils import get_device, load_model

SEEDS = (0,)   # les runs d'un même nom diffèrent par leur seed d'entraînement : c'est elle qui compte
DRS = {'method': 'drs_azadi', 'percentile': 80}
PLAIN = {'method': 'plain'}


def load_models(g_path, d_path, device):
    """(G, D) en mode évaluation ; D vaut None sans chemin."""
    G = load_model(Generator(784).to(device), g_path, device).eval()
    D = load_model(Discriminator(784).to(device), d_path, device).eval() if d_path else None
    return G, D


def measure(G, D, seed, device):
    """(FID, précision, rappel, taux d'acceptation, classes) de 10 000 images d'une seed."""
    x, stats = sample(G, D, 10_000, DRS if D is not None else PLAIN, seed=seed, batch_size=10_000)
    m = evaluate_prof(to_unit_images(x), device, reference='test')
    return m['fid'], m['precision'], m['recall'], stats['acceptance_rate'], '-'.join(map(str, m['classes']))


if __name__ == '__main__':
    device = get_device()
    with open(sys.argv[1], newline='') as f:
        manifest = [(row + [''])[:3] for row in csv.reader(f) if row]
    with open(sys.argv[2], 'w', newline='') as f:
        out = csv.writer(f, lineterminator='\n')
        out.writerow(['name', 'run', 'seed', 'fid', 'precision', 'recall', 'acceptance_rate', 'classes'])
        for name, g_path, d_path in manifest:
            G, D = load_models(g_path, d_path, device)
            for seed in SEEDS:
                row = [name, g_path, seed, *measure(G, D, seed, device)]
                out.writerow(row)
                f.flush()
                print(row[:7], flush=True)
