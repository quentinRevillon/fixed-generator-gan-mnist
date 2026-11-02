"""Génère exactement 10 000 images MNIST 28x28 dans samples/ (interface imposée par le sujet).

Lit le modèle rendu dans checkpoints/ : G.pth, D.pth et sampling.yaml (méthode d'inférence choisie par
le groupe, ex. {method: obrs, K: 10} ; voir sampling.py). Le correcteur lance simplement :
    python generate.py --batch_size 256

Autres utilisations :
    python generate.py --sampling configs/sampling/obrs.yaml        # tester une autre méthode
    python generate.py --checkpoints runs/baseline_v1 --weights best --sampling configs/sampling/plain.yaml
"""
import argparse
import os

import torchvision

from evaluate import WEIGHT_FILES, load_run, to_unit_images
from sampling import sample
from utils import get_device, load_config

N_SAMPLES = 10_000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 10,000 MNIST samples with the trained GAN.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Taille des batchs de génération.')
    parser.add_argument('--checkpoints', default='checkpoints', help='Dossier contenant G.pth et D.pth.')
    parser.add_argument('--weights', choices=sorted(WEIGHT_FILES), default='last', help='Jeu de poids G et D du run.')
    parser.add_argument('--sampling', default='checkpoints/sampling.yaml', help='Méthode d\'échantillonnage (YAML).')
    parser.add_argument('--seed', type=int, default=0, help='Seed : mêmes images à chaque appel.')
    parser.add_argument('--out', default='samples', help='Dossier de sortie des PNG.')
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device.type}')
    G, D = load_run(args.checkpoints, args.weights, device)
    cfg = load_config(args.sampling)
    print(f'Échantillonnage : {cfg}')

    x, stats = sample(G, D, N_SAMPLES, cfg, seed=args.seed, batch_size=args.batch_size)
    os.makedirs(args.out, exist_ok=True)
    for k, image in enumerate(to_unit_images(x)):
        torchvision.utils.save_image(image, os.path.join(args.out, f'{k}.png'))
    print(f"{N_SAMPLES} images dans {args.out}/ | acceptation {stats['acceptance_rate']:.3f} | "
          f"{stats['n_proposals']} propositions | {stats['seconds']:.1f}s")
