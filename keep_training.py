"""KeepTraining d'Azadi et al. 2019 (DRS) : G figé, on continue d'entraîner D avec arrêt anticipé sur la validation.

Figure 1 du papier : "KeepTraining continues training using early stopping on the validation set."
Le papier ne donne ni nombre de pas ni patience : eval_every, patience, max_steps et lr sont nos choix.

Écrit dans le dossier du run : D_<weights>_kt.pth (meilleur D sur la validation) et keep_training.jsonl (historique).

Utilisation :
    python keep_training.py --run runs/baseline_200ep_prof --weights best
    python evaluate.py --run runs/baseline_200ep_prof --weights best_kt --sampling configs/sampling/drs_azadi.yaml
"""
import argparse
import copy
import json
import os

import torch
import torch.nn.functional as F

from data import mnist_split, train_loader
from evaluate import load_run
from sampling import d_logit, generate_from_latents, gaussian_latents, model_device
from utils import get_device, make_deterministic, set_seed


def discriminator_bce(D, x_real, x_fake):
    """Perte BCE de D : les vraies images visent 1, les fausses 0 (calculée sur les logits, plus stable)."""
    real, fake = d_logit(D, x_real), d_logit(D, x_fake)
    return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
            + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))


@torch.no_grad()
def validation_bce(D, x_real, x_fake, batch_size=2000):
    """BCE moyenne sur des images réelles de validation et un lot fixe d'images générées (sur CPU en entrée)."""
    device = model_device(D)
    losses = [discriminator_bce(D, x_real[i:i + batch_size].to(device), x_fake[i:i + batch_size].to(device)).item()
              for i in range(0, min(len(x_real), len(x_fake)), batch_size)]
    return sum(losses) / len(losses)


def keep_training(G, D, loader, x_val, lr=2e-4, eval_every=100, patience=5, max_steps=20_000, seed=0):
    """Entraîne D seul contre G figé ; renvoie (meilleure copie de D sur la validation, historique).

    On s'arrête après `patience` évaluations sans amélioration de la BCE de validation.
    """
    device = model_device(D)
    rng = torch.Generator().manual_seed(seed)
    x_val_fake = generate_from_latents(G, gaussian_latents(len(x_val), rng), 2000)
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    best_loss, best_D, waited, step = validation_bce(D, x_val, x_val_fake), copy.deepcopy(D), 0, 0
    history = [{'step': 0, 'val_bce': best_loss}]
    while step < max_steps and waited < patience:
        for x, _ in loader:
            x_fake = generate_from_latents(G, gaussian_latents(len(x), rng), len(x)).to(device)
            opt.zero_grad()
            discriminator_bce(D, x.to(device), x_fake).backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                loss = validation_bce(D, x_val, x_val_fake)
                history.append({'step': step, 'val_bce': loss})
                improved = loss < best_loss
                best_loss, best_D, waited = (loss, copy.deepcopy(D), 0) if improved else (best_loss, best_D, waited + 1)
            if step >= max_steps or waited >= patience:
                break
    return best_D.eval(), history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keep training D with G frozen (Azadi et al. 2019).')
    parser.add_argument('--run', required=True, help='Dossier du run contenant G et D.')
    parser.add_argument('--weights', choices=['last', 'best'], default='best', help='Poids de départ.')
    parser.add_argument('--batch_size', type=int, default=100, help='Taille des batchs.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)
    make_deterministic()
    G, D = load_run(args.run, args.weights, device)
    x_val = mnist_split('val')[0].flatten(1) * 2 - 1
    best_D, history = keep_training(G, D, train_loader(args.batch_size, args.seed), x_val, seed=args.seed)

    torch.save(best_D.state_dict(), os.path.join(args.run, f'D_{args.weights}_kt.pth'))
    with open(os.path.join(args.run, 'keep_training.jsonl'), 'w') as f:
        f.writelines(json.dumps(row) + '\n' for row in history)
    best = min(history, key=lambda r: r['val_bce'])
    print(f"BCE de validation : départ {history[0]['val_bce']:.4f} | meilleure {best['val_bce']:.4f} au pas {best['step']} "
          f"| arrêt au pas {history[-1]['step']}")
