"""Extrait et met en cache les features (petit CNN et Inception) de toutes les images utilisées par l'audit du FID.

Ne touche à aucun fichier du dépôt : lit evaluate.py / data.py du worktree, écrit dans scratchpad/results/fid/cache.
Lancer depuis le worktree (chemin relatif checkpoints/mnist_classifier.pth) :
    cd wt-comparaison
    DATA=<scratchpad>/data python <scratchpad>/scripts/fid/extract_features.py <scratchpad>

Ensembles (Inception est lent sur CPU, ~4 images/s avec 2 threads et d'autres calculs en parallèle, d'où des
tailles réduites) :
    test_ref   : 2 000 images de MNIST test (référence Inception) ; le CNN voit les 10 000.
    pool       : 1 000 images du split train + 1 000 du split val (vraies images « générées » fictives).
    deg_<nom>  : les 500 premières images de pool dégradées (Inception : bruit 0,3, flou, mélange, effacement ;
                 le CNN a en plus bruit 0,1 et épaississement).
    gan_<nom>  : images de la baseline commune (brutes et remises aux proportions de test) et de RpGAN + R1 + EMA.
Pour le CNN, chaque dégradation est aussi appliquée à 10 000 images de train (N = 10 000).
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torchvision.transforms.functional as TF

from data import mnist_split
from evaluate import generate, inception_features, load_classifier, predict
from model import Generator
from utils import load_model

torch.set_num_threads(2)
DEVICE = torch.device('cpu')
SCRATCH = sys.argv[1]
CACHE = os.path.join(SCRATCH, 'results', 'fid', 'cache')
MODELS = os.path.join(SCRATCH, 'models', 'fid')
N_TEST_INC, N_POOL_HALF, N_DEG_INC, N_GAN_INC = 2000, 1000, 500, 500
INC_DEGRADATIONS = ('noise0.3', 'blur1.0', 'blend', 'erase10')


# --- Dégradations (atomiques) : images (N, 1, 28, 28) dans [0, 1] -> même forme, dans [0, 1] --------------------

def add_noise(x, sigma, seed=0):
    """Bruit gaussien additif d'écart-type sigma, puis écrêtage à [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return (x + sigma * torch.randn(x.shape, generator=g)).clamp(0, 1)


def blur(x, sigma):
    """Flou gaussien (noyau 5x5)."""
    return TF.gaussian_blur(x, kernel_size=5, sigma=sigma)


def blend(x, y, x_other, y_other, seed=0):
    """Chiffre « chimère » : moyenne de chaque image avec une image d'un AUTRE chiffre (trait ambigu, mal formé)."""
    g = torch.Generator().manual_seed(seed)
    partners = []
    for label in y.tolist():
        candidates = (y_other != label).nonzero().squeeze(1)
        partners.append(candidates[torch.randint(len(candidates), (1,), generator=g)].item())
    return 0.5 * x + 0.5 * x_other[partners]


def erase(x, size=10, seed=0):
    """Efface un carré size x size placé au hasard dans la zone centrale 20x20 (trait cassé, morceau manquant)."""
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for i in range(len(x)):
        r, c = torch.randint(4, 24 - size + 1, (2,), generator=g).tolist()
        out[i, :, r:r + size, c:c + size] = 0
    return out


def thicken(x):
    """Épaissit le trait (dilatation 3x3) : change le style, garde le chiffre."""
    return torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)


def degradations(x, y, x_other, y_other):
    """Dictionnaire nom -> images dégradées."""
    return {'noise0.1': add_noise(x, 0.1), 'noise0.3': add_noise(x, 0.3), 'blur1.0': blur(x, 1.0),
            'blend': blend(x, y, x_other, y_other), 'erase10': erase(x), 'thicken': thicken(x)}


# --- Cache --------------------------------------------------------------------------------------------------

def cached(name, compute):
    """Charge CACHE/name.pt s'il existe, sinon le calcule, le sauvegarde et le renvoie."""
    path = os.path.join(CACHE, f'{name}.pt')
    if os.path.exists(path):
        return torch.load(path)
    value = compute()
    torch.save(value, path)
    print('ecrit', name, flush=True)
    return value


def rebalance(images, preds, labels, n):
    """n images dont les chiffres prédits suivent les proportions de `labels`."""
    counts = (torch.bincount(labels, minlength=10).double() / len(labels) * n).round().long()
    return torch.cat([images[preds == d][:counts[d]] for d in range(10)])


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    clf = load_classifier(DEVICE)
    cnn = lambda x: predict(clf, x, DEVICE)[0]
    g = torch.Generator().manual_seed(0)
    x_test, y_test = mnist_split('test')
    x_train, y_train = mnist_split('train')
    x_val, y_val = mnist_split('val')
    i_test = torch.randperm(len(x_test), generator=g)
    i_train = torch.randperm(len(x_train), generator=g)
    i_val = torch.randperm(len(x_val), generator=g)

    # Vraies images : labels et features CNN de tous les splits
    cached('labels', lambda: dict(test=y_test, train=y_train, val=y_val, i_test=i_test, i_train=i_train, i_val=i_val))
    for name, x in (('test', x_test), ('train', x_train), ('val', x_val)):
        cached(f'cnn_{name}', lambda: cnn(x))

    # Dégradations, CNN : 10 000 images de train (indices i_train[:10000])
    xd, yd = x_train[i_train[:10_000]], y_train[i_train[:10_000]]
    xo, yo = x_train[i_train[-5000:]], y_train[i_train[-5000:]]   # partenaires du mélange, hors de l'ensemble
    for name, x in degradations(xd, yd, xo, yo).items():
        cached(f'cnn_deg_{name}', lambda: cnn(x))

    # GAN : 60 000 images (seed 0 ; assez de « 2 » pour la remise en proportion), classes prédites par le CNN
    gans = {}
    for name in ('cmploc_baseline_s0', 'cmploc_rpgan_r1_ema_s0'):
        G = load_model(Generator(784), os.path.join(MODELS, f'{name}_G.pth'), 'cpu').eval()
        gans[name] = generate(G, 60_000, DEVICE, seed=0)
        cached(f'cnn_gan_{name}', lambda: cnn(gans[name]))
    base = gans['cmploc_baseline_s0']
    base_bal = rebalance(base, predict(clf, base, DEVICE)[1], y_test, 10_000)
    assert len(base_bal) == 10_000
    base_bal = base_bal[torch.randperm(10_000, generator=g)]   # mélange : les 1 000 premières gardent les proportions
    cached('cnn_gan_baseline_rebalanced', lambda: cnn(base_bal))

    # Inception (lent) : référence, réservoir, dégradations, GAN
    t_ref = x_test[i_test[:N_TEST_INC]]
    cached('inc_test_ref', lambda: inception_features(t_ref, DEVICE))
    p_train, p_val = x_train[i_train[:N_POOL_HALF]], x_val[i_val[:N_POOL_HALF]]
    cached('inc_pool_train', lambda: inception_features(p_train, DEVICE))
    cached('inc_pool_val', lambda: inception_features(p_val, DEVICE))
    cached('inc_gan_baseline', lambda: inception_features(base[:N_GAN_INC], DEVICE))
    cached('inc_gan_baseline_rebalanced', lambda: inception_features(base_bal[:N_GAN_INC], DEVICE))
    cached('inc_gan_rpgan', lambda: inception_features(gans['cmploc_rpgan_r1_ema_s0'][:N_GAN_INC], DEVICE))
    # les 500 premières images de pool_train = les 500 premières de xd (mêmes indices i_train[:500])
    for name, x in degradations(xd[:N_DEG_INC], yd[:N_DEG_INC], xo, yo).items():
        if name in INC_DEGRADATIONS:
            cached(f'inc_deg_{name}', lambda: inception_features(x, DEVICE))
    print('termine', flush=True)
