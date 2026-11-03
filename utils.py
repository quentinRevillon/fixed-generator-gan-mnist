"""Utilitaires partagés : pas d'entraînement D / G, config, seed, déterminisme, device, checkpoints.

Utilisation :
    from utils import load_config, set_seed, make_deterministic, get_device, load_model
    cfg = load_config('configs/baseline.yaml')
    set_seed(cfg['seed']); make_deterministic()
    G = load_model(Generator(784).to(device), 'runs/baseline_v1/G.pth', device)
"""
import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# cuBLAS n'est déterministe qu'avec cet espace de travail ; doit être défini avant la première opération CUDA.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def D_logits(D, x):
    """Sortie de D avant la sigmoïde finale (model.py reste inchangé)."""
    D = unwrap(D)
    for fc in (D.fc1, D.fc2, D.fc3):
        x = F.leaky_relu(fc(x), 0.2)
    return D.fc4(x)


def zero_centered_gp(samples, logits):
    """||grad_x D(x)||^2 moyenné sur le batch : R1 sur les vraies images, R2 sur les fausses."""
    grad, = torch.autograd.grad(outputs=logits.sum(), inputs=samples, create_graph=True)
    return grad.pow(2).sum(dim=1).mean()


def D_train_rpgan(x, G, D, D_optimizer, device, r1_gamma, r2_gamma):
    """Un pas de D avec RpGAN + R1 + R2 (R3GAN, Huang et al. 2024) ; renvoie (loss_D, D(x) moyen, D(G(z)) moyen)."""
    D.zero_grad()
    x_real = x.to(device).requires_grad_(r1_gamma > 0)
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z).detach().requires_grad_(r2_gamma > 0)
    real_logits = D_logits(D, x_real)
    fake_logits = D_logits(D, x_fake)

    # paire relativiste : D veut D(réel) > D(faux)
    D_loss = F.softplus(fake_logits - real_logits).mean()
    if r1_gamma > 0:
        D_loss = D_loss + r1_gamma / 2 * zero_centered_gp(x_real, real_logits)
    if r2_gamma > 0:
        D_loss = D_loss + r2_gamma / 2 * zero_centered_gp(x_fake, fake_logits)
    D_loss.backward()
    D_optimizer.step()

    # D(x) et D(G(z)) rapportés en probabilités, comme pour la baseline
    return D_loss.item(), torch.sigmoid(real_logits).mean().item(), torch.sigmoid(fake_logits).mean().item()


def G_train_rpgan(x, G, D, G_optimizer, device):
    """Un pas de G avec RpGAN : G veut D(faux) > D(réel) ; renvoie loss_G."""
    G.zero_grad()
    z = torch.randn(x.shape[0], 100, device=device)
    fake_logits = D_logits(D, G(z))
    with torch.no_grad():
        real_logits = D_logits(D, x.to(device))
    G_loss = F.softplus(real_logits - fake_logits).mean()
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()


def D_train(x, G, D, D_optimizer, criterion, device, loss='bce', r1_gamma=0., r2_gamma=0.):
    """Un pas d'optimisation du discriminateur ; renvoie (loss_D, D(x) moyen, D(G(z)) moyen)."""
    if loss == 'rpgan':
        return D_train_rpgan(x, G, D, D_optimizer, device, r1_gamma, r2_gamma)
    D.zero_grad()

    # vraies images
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # fausses images
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # rétropropagation : seuls les paramètres de D sont mis à jour
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return D_loss.data.item(), D_real_score.mean().item(), D_fake_score.mean().item()


def G_train(x, G, D, G_optimizer, criterion, device, loss='bce'):
    """Un pas d'optimisation du générateur (loss non saturante) ; renvoie loss_G."""
    if loss == 'rpgan':
        return G_train_rpgan(x, G, D, G_optimizer, device)
    G.zero_grad()
    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # rétropropagation : seuls les paramètres de G sont mis à jour
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()


def load_config(path):
    """Lit un fichier YAML et renvoie un dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit():
    """Hash court du commit courant du repo ('unknown' hors d'un checkout git)."""
    out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip() or 'unknown'


def set_seed(seed):
    """Fixe les générateurs globaux de python, numpy et torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_deterministic():
    """Demande à PyTorch des algorithmes déterministes (avertissement, pas d'erreur, si un op ne l'est pas)."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def rng_state():
    """États des générateurs globaux torch (CPU et CUDA), pour une reprise exacte."""
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    """Restaure des états produits par rng_state()."""
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def environment(device):
    """Versions et matériel, enregistrés avec chaque run (Mac MPS et GPU CUDA ne donnent pas les mêmes poids)."""
    name = torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor() or device.type
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': f'{device.type}: {name}'}


def get_device():
    """Meilleur device disponible : CUDA, puis MPS (Apple), puis CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def unwrap(model):
    """Renvoie le module sous-jacent d'un DataParallel (ou le modèle lui-même)."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_ckpt(path, G, D, G_opt, D_opt, **extra):
    """Sauvegarde l'état complet d'entraînement (modèles, optimiseurs, RNG globaux + champs `extra`)."""
    torch.save({'G': unwrap(G).state_dict(), 'D': unwrap(D).state_dict(),
                'G_opt': G_opt.state_dict(), 'D_opt': D_opt.state_dict(), 'rng': rng_state(), **extra}, path)


def load_ckpt(path, G, D, G_opt, D_opt, device):
    """Restaure modèles, optimiseurs et RNG globaux sauvegardés par save_ckpt ; renvoie le dict complet."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(G).load_state_dict(ckpt['G'])
    unwrap(D).load_state_dict(ckpt['D'])
    G_opt.load_state_dict(ckpt['G_opt'])
    D_opt.load_state_dict(ckpt['D_opt'])
    set_rng_state(ckpt['rng'])
    return ckpt


def load_model(model, path, device):
    """Charge des poids (.pth, ou dossier contenant G.pth) en retirant les préfixes 'module.' de DataParallel."""
    path = path if path.endswith('.pth') else os.path.join(path, 'G.pth')
    state = torch.load(path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    return model
