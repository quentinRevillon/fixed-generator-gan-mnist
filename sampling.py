"""Inférence : comment tirer des images d'un couple (G, D) déjà entraîné, sans toucher aux poids de G.

Toutes les méthodes ont la même signature :
    sample_<method>(G, D, n, rng, batch_size, **params) -> (x, stats)
        x     : exactement n images (n, 784) dans [-1, 1] (sortie tanh de G), sur CPU
        stats : dict avec au moins 'n_proposals' et 'acceptance_rate' (+ paramètres propres à la méthode)

    méthode      | besoin | rôle dans l'étude
    -------------|--------|-------------------------------------------------------------
    'plain'      | G      | référence : x = G(z), z ~ N(0, I)
    'truncation' | G      | témoin : z tronqué à [-psi, psi] (Brock et al. 2019)
    'drs'        | G, D   | témoin : rejet de Azadi et al. 2019 (slides Def. 3.1)
    'obrs'       | G, D   | méthode étudiée : rejet optimal à budget K (Verine et al. 2024, slides Thm 3.2)

Utilisation :
    from sampling import sample
    x, stats = sample(G, D, 10_000, {'method': 'obrs', 'K': 10}, seed=0)
    stats -> {'method': 'obrs', 'n_proposals': ..., 'acceptance_rate': ..., 'seconds': ..., 'tau_K': ...}

Ajouter une méthode : écrire sample_xxx(G, D, n, rng, batch_size, **params), l'ajouter à SAMPLERS
(en bas du fichier) et créer configs/sampling/xxx.yaml. Rien d'autre à modifier.

Convention : tout le hasard est tiré sur CPU avec un torch.Generator dédié (`rng`), puis envoyé sur le
device du modèle. Même seed => mêmes images, quel que soit le device.
"""
import math
import time

import torch
import torch.nn.functional as F

Z_DIM = 100


# --- Espace latent : méthodes sans discriminateur -------------------------------------------------------------

def model_device(model):
    """Device sur lequel se trouvent les poids du modèle."""
    return next(model.parameters()).device


def gaussian_latents(n, rng):
    """n latents z ~ N(0, I_100), forme (n, 100), sur CPU."""
    return torch.randn(n, Z_DIM, generator=rng)


def truncated_latents(n, psi, rng):
    """n latents de loi normale tronquée coordonnée par coordonnée à [-psi, psi] (truncation trick).

    Tirage exact par inversion de la fonction de répartition : u ~ U(Phi(-psi), Phi(psi)), z = Phi^-1(u).
    Équivaut à rééchantillonner chaque |z_i| > psi, sans boucle ni rejet.
    """
    lo = torch.special.ndtr(torch.tensor(-psi, dtype=torch.float64))
    u = lo + (1 - 2 * lo) * torch.rand(n, Z_DIM, generator=rng, dtype=torch.float64)
    return torch.special.ndtri(u).float()


@torch.no_grad()
def generate_from_latents(G, z, batch_size):
    """Images G(z) (n, 784) dans [-1, 1], calculées par batchs sur le device de G, renvoyées sur CPU."""
    device = model_device(G)
    return torch.cat([G(z[i:i + batch_size].to(device)).cpu() for i in range(0, len(z), batch_size)])


def sample_plain(G, D, n, rng, batch_size=10_000):
    """Échantillonnage standard x = G(z), z ~ N(0, I) : la référence de toutes les comparaisons."""
    x = generate_from_latents(G, gaussian_latents(n, rng), batch_size)
    return x, {'n_proposals': n, 'acceptance_rate': 1.0}


def sample_truncation(G, D, n, rng, batch_size=10_000, psi=1.0):
    """Truncation trick : x = G(z), z normale tronquée à [-psi, psi]^100. psi petit => précision ↑, rappel ↓."""
    x = generate_from_latents(G, truncated_latents(n, psi, rng), batch_size)
    return x, {'n_proposals': n, 'acceptance_rate': 1.0, 'psi': psi}


# --- Le discriminateur comme estimateur du rapport de densités -----------------------------------------------
# Si D est optimal, D(x) = p(x) / (p(x) + p̂(x)), donc r(x) = p(x) / p̂(x) = D / (1 - D) = exp(d(x)), où d est
# le logit de D (sortie AVANT la sigmoïde). On travaille en log : log r(x) = d(x). Calculer log(D / (1 - D))
# après la sigmoïde est instable (D arrondi à 1.0 en float32 => logit infini).

def d_logit(D, x):
    """Logit d(x) du Discriminator de model.py (même calcul que D.forward, sans la sigmoïde finale).

    Si l'architecture de D change (volet Entraînement), adapter cette fonction en conséquence.
    """
    h = F.leaky_relu(D.fc1(x), 0.2)
    h = F.leaky_relu(D.fc2(h), 0.2)
    h = F.leaky_relu(D.fc3(h), 0.2)
    return D.fc4(h).squeeze(1)


@torch.no_grad()
def batched_logits(D, x, batch_size):
    """d(x) pour x (N, 784) sur CPU, calculé par batchs sur le device de D ; renvoie (N,) sur CPU."""
    device = model_device(D)
    return torch.cat([d_logit(D, x[i:i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)])


def propose(G, D, n, rng, batch_size):
    """n propositions x = G(z), z ~ N(0, I), et leurs log-ratios estimés log r(x) = d(x)."""
    x = generate_from_latents(G, gaussian_latents(n, rng), batch_size)
    return x, batched_logits(D, x, batch_size)


# --- Rejet piloté par D : OBRS (méthode étudiée) et DRS (témoin) ---------------------------------------------
# On propose x ~ p̂ et on l'accepte avec probabilité a(x) ; les images gardées suivent p̃_a ∝ p̂ · a.
# Le budget K = 1 / E_p̂[a] est le nombre moyen de propositions par image gardée (le coût).
#   OBRS : a(x) = min(1, r(x) c_K / M) = min(1, exp(d(x) - tau_K)), avec E_p̂[a] = 1/K. Optimal pour toute
#          f-divergence à budget fixé ; Prop 3.3 : précision multipliée par K (plafonnée à 1), rappel conservé.
#   DRS  : a(x) = r / (r (1 - e^gamma) + M e^gamma) ; gamma est calibré ici pour le même budget K.

def obrs_threshold(log_r, K, n_iter=100):
    """tau_K tel que mean(min(1, exp(log_r - tau_K))) = 1/K sur le pool log_r (bissection).

    tau_K est plafonné à max(log_r) (c_K >= 1) : si le budget K dépasse M, OBRS devient le rejet exact.
    """
    log_r = log_r.double()
    lo, hi = log_r.min().item(), log_r.max().item() + math.log(K)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if obrs_accept_prob(log_r, mid).mean().item() > 1 / K else (lo, mid)
    return min((lo + hi) / 2, log_r.max().item())


def obrs_accept_prob(log_r, tau):
    """a_OBRS = min(1, exp(log_r - tau))."""
    return torch.exp((log_r - tau).clamp(max=0))


def drs_accept_prob(log_r, log_M, gamma):
    """a_DRS = r / (r (1 - e^gamma) + M e^gamma), calculé avec s = r / M = exp(log_r - log_M).

    Une proposition plus haute que le max du pool (s >= 1) est acceptée (a = 1).
    """
    s = torch.exp(log_r.double() - log_M)
    a = s / (s + (1 - s) * math.exp(gamma))
    return torch.where(s >= 1, torch.ones_like(a), a).to(log_r.dtype)


def drs_gamma(log_r, log_M, K, n_iter=100):
    """gamma tel que mean(a_DRS) = 1/K sur le pool log_r (bissection ; a_DRS décroît avec gamma)."""
    lo, hi = -50.0, 50.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if drs_accept_prob(log_r, log_M, mid).mean().item() > 1 / K else (lo, mid)
    return (lo + hi) / 2


def rejection_sample(G, D, n, rng, batch_size, accept_prob):
    """Propose par batchs, accepte chaque x avec probabilité accept_prob(log_r), jusqu'à n images exactement.

    Renvoie (x (n, 784) dans [-1, 1], nombre total de propositions, nombre total d'acceptations).
    """
    kept, n_proposals, n_accepted = [], 0, 0
    while n_accepted < n:
        x, log_r = propose(G, D, batch_size, rng, batch_size)
        keep = torch.rand(batch_size, generator=rng) < accept_prob(log_r)
        kept.append(x[keep])
        n_proposals, n_accepted = n_proposals + batch_size, n_accepted + int(keep.sum())
    return torch.cat(kept)[:n], n_proposals, n_accepted


def calibration_pool(G, D, rng, n_calib, batch_size):
    """log r(x) d'un pool de n_calib propositions, utilisé seulement pour calibrer tau_K ou (M, gamma)."""
    return propose(G, D, n_calib, rng, batch_size)[1]


def sample_obrs(G, D, n, rng, batch_size=10_000, K=10, n_calib=50_000):
    """OBRS à budget K : calibre tau_K sur un pool, puis rejet avec a_OBRS."""
    tau = obrs_threshold(calibration_pool(G, D, rng, n_calib, batch_size), K)
    x, n_prop, n_acc = rejection_sample(G, D, n, rng, batch_size, lambda log_r: obrs_accept_prob(log_r, tau))
    return x, {'n_proposals': n_prop, 'acceptance_rate': n_acc / n_prop, 'K': K, 'tau_K': tau, 'n_calib': n_calib}


def sample_drs(G, D, n, rng, batch_size=10_000, K=10, n_calib=50_000):
    """DRS au même budget K : M = max du pool, gamma calibré pour E[a_DRS] = 1/K, puis rejet avec a_DRS."""
    pool = calibration_pool(G, D, rng, n_calib, batch_size)
    log_M = pool.max().item()
    gamma = drs_gamma(pool, log_M, K)
    x, n_prop, n_acc = rejection_sample(G, D, n, rng, batch_size, lambda log_r: drs_accept_prob(log_r, log_M, gamma))
    return x, {'n_proposals': n_prop, 'acceptance_rate': n_acc / n_prop, 'K': K, 'gamma': gamma, 'log_M': log_M,
               'n_calib': n_calib}


# --- Registre et point d'entrée -------------------------------------------------------------------------------

SAMPLERS = {
    'plain': sample_plain,
    'truncation': sample_truncation,
    'drs': sample_drs,
    'obrs': sample_obrs,
}


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
