"""KID et FID par chiffre, dans les features Inception (et, pour comparer, FID par chiffre du petit CNN).

KID (Bińkowski et al. 2018) : MMD² sans biais avec le noyau polynomial k(x, y) = (x·y / d + 1)^3, sur les mêmes features
que le FID. Sans biais : sa valeur ne dépend pas du nombre d'images, et vaut 0 en moyenne entre deux échantillons réels.
Global : moyenne ± écart-type sur 100 sous-ensembles de 1 000 images (convention du papier), rapporté × 1000.
FID par chiffre : on génère 100 000 images, le petit CNN les classe, on garde 1 000 images de chaque chiffre prédit,
et on compare chacune aux images de ce chiffre dans MNIST test (vrais labels, 892 à 1 000). Même nombre d'images pour
tous les modèles, donc même biais : les valeurs se comparent entre modèles, pas aux FID à 10 000 images.
Plancher : la ligne « real val » (vraies images de val, classes prédites par le CNN comme pour les GAN).

Manifeste : CSV sans en-tête, une ligne par modèle : nom,chemin_de_G[,chemin_de_D] (avec D : DRS au percentile 80).
Le chemin « val » remplace G par les vraies images de val.
Usage, depuis la racine du repo (branche noe/training), sur GPU :
    PYTHONPATH=. python scripts/kid_perclass.py <manifeste.csv> <sortie.csv>
"""
import csv
import sys

import torch

from data import mnist_split
from evaluate import fid, inception_features, load_classifier, predict, to_unit_images
from model import Discriminator, Generator
from sampling import sample
from utils import get_device, load_model

N_GLOBAL, N_POOL, PER_DIGIT = 10_000, 100_000, 1000
KID_SUBSETS, KID_SIZE = 100, 1000
DRS = {'method': 'drs_azadi', 'percentile': 80}
PLAIN = {'method': 'plain'}


def poly_kernel(a, b):
    """Noyau du KID : (a·b / d + 1)^3, en double précision."""
    return (a @ b.T / a.shape[1] + 1) ** 3


def mmd2_unbiased(x, y):
    """MMD² sans biais entre deux échantillons de features (tailles quelconques, au moins 2 chacun)."""
    x, y = x.double(), y.double()
    kxx, kyy = poly_kernel(x, x), poly_kernel(y, y)
    n, m = len(x), len(y)
    return ((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
            - 2 * poly_kernel(x, y).mean()).item()


def kid(f_real, f_fake, seed=0):
    """(moyenne, écart-type) du MMD² sans biais sur KID_SUBSETS sous-ensembles de KID_SIZE images."""
    rng = torch.Generator().manual_seed(seed)
    vals = torch.tensor([mmd2_unbiased(f_real[torch.randperm(len(f_real), generator=rng)[:KID_SIZE]],
                                       f_fake[torch.randperm(len(f_fake), generator=rng)[:KID_SIZE]])
                         for _ in range(KID_SUBSETS)])
    return vals.mean().item(), vals.std().item()


def per_digit_indices(labels):
    """Indices des PER_DIGIT premières images de chaque chiffre (moins si le chiffre est plus rare)."""
    return [torch.nonzero(labels == c).view(-1)[:PER_DIGIT] for c in range(10)]


def model_images(g_path, d_path, n, seed, device):
    """n images (n, 1, 28, 28) dans [0, 1] : vraies images de val si g_path == 'val', sinon G (avec DRS si D)."""
    if g_path == 'val':
        x = mnist_split('val')[0]
        return x[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))][:n]
    G = load_model(Generator(784).to(device), g_path, device).eval()
    D = load_model(Discriminator(784).to(device), d_path, device).eval() if d_path else None
    return to_unit_images(sample(G, D, n, DRS if D is not None else PLAIN, seed=seed, batch_size=10_000)[0])


def real_reference(device, clf):
    """Features Inception de MNIST test, et pour chaque chiffre : (features Inception, features CNN) de ses images (au plus 1 000)."""
    x, y = mnist_split('test')
    f_inc = inception_features(x, device)
    f_cnn = predict(clf, x, device)[0]
    return f_inc, [(f_inc[i], f_cnn[i]) for i in per_digit_indices(y)]


def measure(name, g_path, d_path, ref, device, clf):
    """Lignes de sortie d'un modèle : une ligne « all » (10 000 images), une par chiffre, une « mean »."""
    f_inc_real, real_digits = ref
    f_inc = inception_features(model_images(g_path, d_path, N_GLOBAL, 0, device), device)
    k_mean, k_std = kid(f_inc_real, f_inc)
    rows = [dict(name=name, g=g_path, digit='all', n=len(f_inc), fid_inception=fid(f_inc_real, f_inc),
                 kid_inception_x1000=1000 * k_mean, kid_std_x1000=1000 * k_std, fid_cnn='')]
    pool = model_images(g_path, d_path, N_POOL, 1, device)
    f_cnn_pool, labels = predict(clf, pool, device)
    for c, idx in enumerate(per_digit_indices(labels)):
        f_inc_c = inception_features(pool[idx], device)
        r_inc, r_cnn = real_digits[c]
        rows.append(dict(name=name, g=g_path, digit=c, n=len(idx), fid_inception=fid(r_inc, f_inc_c),
                         kid_inception_x1000=1000 * mmd2_unbiased(r_inc, f_inc_c), kid_std_x1000='',
                         fid_cnn=fid(r_cnn, f_cnn_pool[idx])))
    digits = rows[1:]
    rows.append(dict(name=name, g=g_path, digit='mean', n=sum(r['n'] for r in digits),
                     **{k: sum(r[k] for r in digits) / 10 for k in ('fid_inception', 'kid_inception_x1000', 'fid_cnn')},
                     kid_std_x1000=''))
    return rows


def main(manifest_path, out_path):
    """Mesure chaque ligne du manifeste et écrit les lignes au fur et à mesure."""
    device = get_device()
    clf = load_classifier(device)
    ref = real_reference(device, clf)
    with open(manifest_path, newline='') as f:
        manifest = [(row + [''])[:3] for row in csv.reader(f) if row]
    fields = ['name', 'g', 'digit', 'n', 'fid_inception', 'kid_inception_x1000', 'kid_std_x1000', 'fid_cnn']
    with open(out_path, 'w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        out.writeheader()
        for name, g_path, d_path in manifest:
            for row in measure(name, g_path, d_path, ref, device, clf):
                out.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
