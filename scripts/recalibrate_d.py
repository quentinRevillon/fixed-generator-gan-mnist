"""Recalibre le discriminateur rendu pour le rejet OBRS (Azadi et al. 2019, §4.2 ; rapport, Sec. 5).

En fin d'entraînement RpGAN + R1, D ne sépare presque plus réels et générés (AUC 0,57) : son logit trie mal.
On garde ses couches fc1-fc3 et on remplace seulement fc4 par une régression logistique ajustée sur les 256
features de fc3 : 10 000 images réelles de validation contre 10 000 images de G (latents de la seed 100).
L'architecture de D est inchangée : le D recalibré se charge comme l'original et sampling.py l'utilise tel quel.

Usage, depuis la racine du repo :
    python scripts/recalibrate_d.py runs/cmploc_rpgan_r1_ema_s1 checkpoints/D.pth
    # lit <run>/G.pth et <run>/D.pth, écrit le D recalibré ; checkpoints/sampling.yaml : {method: obrs, K: 3}
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import mnist_split  # noqa: E402
from evaluate import load_run  # noqa: E402
from sampling import gaussian_latents, generate_from_latents  # noqa: E402
from utils import get_device  # noqa: E402

N = 10_000
FIT_SEED, AUC_SEED = 100, 200


def real_images(split):
    """Images réelles du split au format de sortie de G : (n, 784) dans [-1, 1]."""
    return mnist_split(split)[0].view(-1, 784) * 2 - 1


def fake_images(G, seed):
    """N images G(z) (N, 784) dans [-1, 1], latents tirés avec cette seed."""
    return generate_from_latents(G, gaussian_latents(N, torch.Generator().manual_seed(seed)), N)


@torch.no_grad()
def fc3_features(D, x):
    """Entrée de la dernière couche de D (sortie de fc3 après leaky ReLU), en numpy."""
    h = x.to(D.fc1.weight.device)
    for fc in (D.fc1, D.fc2, D.fc3):
        h = F.leaky_relu(fc(h), 0.2)
    return h.cpu().numpy()


def fit_head(D, x_real, x_fake):
    """Régression logistique réels (1) contre générés (0) sur les features de fc3."""
    h = np.concatenate([fc3_features(D, x_real), fc3_features(D, x_fake)])
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))]
    return LogisticRegression(C=1.0, max_iter=3000).fit(h, y)


def set_head(D, head):
    """Remplace les poids de fc4 de D (en place) par ceux de la régression logistique."""
    device = D.fc4.weight.device
    D.fc4.weight.data = torch.as_tensor(head.coef_, dtype=torch.float32, device=device)
    D.fc4.bias.data = torch.as_tensor(head.intercept_, dtype=torch.float32, device=device)


@torch.no_grad()
def auc(D, x_real, x_fake):
    """AUC du logit de D pour séparer réels et générés (0,5 = hasard, 1 = parfait)."""
    x = torch.cat([x_real, x_fake]).to(D.fc1.weight.device)
    d = D.fc4(torch.from_numpy(fc3_features(D, x)).to(x.device)).squeeze(1).cpu().numpy()
    return roc_auc_score(np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))], d)


if __name__ == '__main__':
    run_dir, out_path = sys.argv[1], sys.argv[2]
    G, D = load_run(run_dir, 'last', get_device())
    x_test, x_check = real_images('test'), fake_images(G, AUC_SEED)
    before = auc(D, x_test, x_check)
    set_head(D, fit_head(D, real_images('val'), fake_images(G, FIT_SEED)))
    print(f'AUC de D sur test : {before:.3f} -> {auc(D, x_test, x_check):.3f} (recalibré)')
    torch.save(D.state_dict(), out_path)
    print(f'D recalibré écrit dans {out_path}')
