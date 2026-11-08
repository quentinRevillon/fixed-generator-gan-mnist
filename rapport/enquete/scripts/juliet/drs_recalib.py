"""DRS avec D recalibré (Azadi et al. 2019, §4.2), mesuré comme la plateforme : FID Inception, P/R VGG, plus le petit CNN.

Recalibration : la dernière couche de D est remplacée par une régression logistique sur les 256 sorties de fc3,
ajustée sur 10 000 images réelles de val contre 10 000 images de G (latents de la seed 100).
Pour chaque modèle, chaque variante et chaque tirage : 10 000 images acceptées, mesurées contre MNIST test,
avec le taux d'acceptation, le temps total de génération (propositions + notes de D + tri) et l'AUC de D sur test.

Usage, depuis la racine du repo (branche noe/training), sur GPU :
    PYTHONPATH=. python scripts/drs_recalib.py results/drs_recalib_test.csv
"""
import copy
import csv
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import mnist_split
from evaluate import evaluate, evaluate_prof, to_unit_images
from model import Discriminator, Generator
from sampling import batched_logits, gaussian_latents, generate_from_latents, sample
from utils import get_device, load_model

N = 10_000
DRAWS = (0, 1)
MODELS = {f'{c}_s{s}': (f'runs/cmploc_{c}_s{s}/G.pth', f'runs/cmploc_{c}_s{s}/D.pth')
          for c in ('baseline', 'rpgan_r1_ema') for s in range(3)}
VARIANTS = [('plain', False, {'method': 'plain'}),
            ('drs_p80', False, {'method': 'drs_azadi', 'percentile': 80}),
            ('drs_p80_recal', True, {'method': 'drs_azadi', 'percentile': 80}),
            ('obrs_K3', False, {'method': 'obrs', 'K': 3}),
            ('obrs_K3_recal', True, {'method': 'obrs', 'K': 3})]


def real(split):
    """Images réelles du split au format de sortie de G : (n, 784) dans [-1, 1], sur CPU (convention de sampling.py)."""
    return mnist_split(split)[0].view(-1, 784) * 2 - 1


def fake(G, seed):
    """N images G(z) (N, 784) dans [-1, 1], latents tirés avec cette seed."""
    return generate_from_latents(G, gaussian_latents(N, torch.Generator().manual_seed(seed)), N)


@torch.no_grad()
def fc3_features(D, x):
    """Sortie de fc3 de D (après leaky ReLU) : l'entrée de sa dernière couche, calculée sur le device de D."""
    h = x.to(D.fc1.weight.device)
    for fc in (D.fc1, D.fc2, D.fc3):
        h = torch.nn.functional.leaky_relu(fc(h), 0.2)
    return h


def recalibrate(D, x_real, x_fake):
    """Copie de D dont la dernière couche est une régression logistique réels contre G sur les features de fc3."""
    h = torch.cat([fc3_features(D, x_real), fc3_features(D, x_fake)]).cpu().numpy()
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))]
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(h, y)
    D2 = copy.deepcopy(D)
    D2.fc4.weight.data = torch.as_tensor(lr.coef_[0], dtype=torch.float32, device=D.fc4.weight.device).view(1, -1)
    D2.fc4.bias.data = torch.as_tensor([lr.intercept_[0]], dtype=torch.float32, device=D.fc4.weight.device)
    return D2


def auc(D, x_real, x_fake):
    """AUC de D pour séparer réels et faux (0,5 = hasard)."""
    d = torch.cat([batched_logits(D, x_real, N), batched_logits(D, x_fake, N)]).cpu().numpy()
    return roc_auc_score(np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))], d)


def timed_sample(G, D, cfg, seed, device):
    """(images, stats, secondes) pour N images acceptées."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    x, stats = sample(G, D, N, cfg, seed=seed, batch_size=N)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return x, stats, time.perf_counter() - t


def main(out_path):
    device = get_device()
    x_val, x_test = real('val'), real('test')
    fields = ['model', 'variant', 'draw', 'fid', 'precision', 'recall', 'fid_cnn', 'precision_cnn', 'recall_cnn',
              'acceptance_rate', 'seconds', 'auc', 'classes']
    with open(out_path, 'w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        out.writeheader()
        for name, (g_path, d_path) in MODELS.items():
            G = load_model(Generator(784).to(device), g_path, device).eval()
            D = load_model(Discriminator(784).to(device), d_path, device).eval()
            D_recal = recalibrate(D, x_val, fake(G, 100))
            x_auc = fake(G, 200)
            aucs = {False: auc(D, x_test, x_auc), True: auc(D_recal, x_test, x_auc)}
            for label, use_recal, cfg in VARIANTS:
                for draw in DRAWS:
                    x, stats, seconds = timed_sample(G, D_recal if use_recal else D, cfg, draw, device)
                    images = to_unit_images(x)
                    m, c = evaluate_prof(images, device, reference='test'), evaluate(images, device, reference='test')
                    row = dict(model=name, variant=label, draw=draw, fid=m['fid'], precision=m['precision'],
                               recall=m['recall'], fid_cnn=c['fid'], precision_cnn=c['precision'], recall_cnn=c['recall'],
                               acceptance_rate=stats['acceptance_rate'], seconds=seconds, auc=aucs[use_recal],
                               classes='-'.join(map(str, m['classes'])))
                    out.writerow(row)
                    f.flush()
                    print(row, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
