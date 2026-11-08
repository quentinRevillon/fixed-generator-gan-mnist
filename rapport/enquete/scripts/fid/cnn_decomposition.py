"""FID du petit CNN décomposé : terme de moyenne / de covariance, et FID moyen « par chiffre » (qualité à chiffre fixé).

Lancer depuis le worktree, après extract_features.py :
    python <scratchpad>/scripts/fid/cnn_decomposition.py <scratchpad>
Sortie : <scratchpad>/results/fid/cnn_decomposition.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

import torch
import torch.nn.functional as F

from audit_imbalance import fid_terms
from evaluate import fid, load_classifier

torch.set_num_threads(2)
SCRATCH = sys.argv[1]
CACHE = os.path.join(SCRATCH, 'results', 'fid', 'cache')
OUT = os.path.join(SCRATCH, 'results', 'fid', 'cnn_decomposition.csv')


def load(name):
    """Objet en cache."""
    return torch.load(os.path.join(CACHE, f'{name}.pt'))


@torch.no_grad()
def classes(features):
    """Chiffre prédit par le classifieur à partir des features (avant ReLU) en cache."""
    return load_classifier(torch.device('cpu')).out(F.relu(features)).argmax(1)


def per_class_fid(f_real, y_real, f_fake, y_fake, digits, count=500):
    """Moyenne sur `digits` de FID(count images générées du chiffre d, toutes les images réelles de test du chiffre d).

    Même nombre d'images générées par chiffre pour tous les ensembles : le biais lié à N est le même partout.
    """
    values = [fid(f_real[y_real == d], f_fake[y_fake == d][:count]) for d in digits]
    return sum(values) / len(values)


if __name__ == '__main__':
    lab = load('labels')
    f_test, y_test = load('cnn_test'), lab['test']
    rest = lab['i_train'][10_000:]
    sets = {'reel (train 10k)': (load('cnn_train')[rest[:10_000]], lab['train'][rest[:10_000]]),
            'baseline': (load('cnn_gan_cmploc_baseline_s0')[:10_000], None),
            'baseline remise en proportion': (load('cnn_gan_baseline_rebalanced'), None),
            'rpgan_r1_ema': (load('cnn_gan_cmploc_rpgan_r1_ema_s0')[:10_000], None)}
    rows = []
    for name, (f, y) in sets.items():
        y = classes(f) if y is None else y   # vraies images : vrais labels ; GAN : chiffre prédit
        mean_term, cov_term = fid_terms(f_test, f)
        no_two = [d for d in range(10) if d != 2]   # la baseline brute a moins de 500 « 2 »
        rows.append(dict(set=name, fid=round(mean_term + cov_term, 2), mean_term=round(mean_term, 2),
                         cov_term=round(cov_term, 2),
                         per_class_fid_9_digits=round(per_class_fid(f_test, y_test, f, y, no_two), 2),
                         per_class_fid_digit_2=round(fid(f_test[y_test == 2], f[y == 2][:500]), 2) if (y == 2).sum() >= 500 else ''))
        print(rows[-1], flush=True)
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
