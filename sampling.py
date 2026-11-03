"""Comment tirer des images d'un couple (G, D) déjà entraîné, sans toucher aux poids de G.

Toutes les méthodes ont la même signature :
    sample_<method>(G, D, n, rng, batch_size, **params) -> (x, stats)
        x     : exactement n images (n, 784) dans [-1, 1] (sortie tanh de G), sur CPU
        stats : dict avec au moins 'n_proposals' et 'acceptance_rate' (+ paramètres propres à la méthode)

Sur cette branche, seule la référence existe : 'plain', x = G(z), z ~ N(0, I). Les méthodes d'inférence sont sur la
branche quentin/inference.

Utilisation :
    from sampling import sample
    x, stats = sample(G, D, 10_000, {'method': 'plain'}, seed=0)
    stats -> {'method': 'plain', 'n_proposals': ..., 'acceptance_rate': ..., 'seconds': ...}

Ajouter une méthode : écrire sample_xxx(G, D, n, rng, batch_size, **params), l'ajouter à SAMPLERS
(en bas du fichier) et créer configs/sampling/xxx.yaml. Rien d'autre à modifier.

Convention : tout le hasard est tiré sur CPU avec un torch.Generator dédié (`rng`), puis envoyé sur le
device du modèle. Même seed => mêmes images, quel que soit le device.
"""
import time

import torch

Z_DIM = 100


def model_device(model):
    """Device sur lequel se trouvent les poids du modèle."""
    return next(model.parameters()).device


def gaussian_latents(n, rng):
    """n latents z ~ N(0, I_100), forme (n, 100), sur CPU."""
    return torch.randn(n, Z_DIM, generator=rng)


@torch.no_grad()
def generate_from_latents(G, z, batch_size):
    """Images G(z) (n, 784) dans [-1, 1], calculées par batchs sur le device de G, renvoyées sur CPU."""
    device = model_device(G)
    return torch.cat([G(z[i:i + batch_size].to(device)).cpu() for i in range(0, len(z), batch_size)])


def sample_plain(G, D, n, rng, batch_size=10_000):
    """Échantillonnage standard x = G(z), z ~ N(0, I) : la référence de toutes les comparaisons."""
    x = generate_from_latents(G, gaussian_latents(n, rng), batch_size)
    return x, {'n_proposals': n, 'acceptance_rate': 1.0}


SAMPLERS = {'plain': sample_plain}


def sample(G, D, n, cfg, seed=0, batch_size=10_000):
    """Tire n images avec la méthode cfg['method'] et ses paramètres (reste de cfg) ; ajoute méthode et durée."""
    params = {k: v for k, v in cfg.items() if k != 'method'}
    G.eval()
    if D is not None:
        D.eval()
    rng = torch.Generator().manual_seed(seed)
    t0 = time.time()
    x, stats = SAMPLERS[cfg['method']](G, D, n, rng, batch_size=batch_size, **params)
    return x, {'method': cfg['method'], **stats, 'seconds': time.time() - t0}
