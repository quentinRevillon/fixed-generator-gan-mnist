"""Entraîne le GAN décrit par une config YAML (boucle complète train(cfg)).

Utilisation :
    python train.py --config configs/baseline.yaml            # sorties dans runs/<name>/
    python train.py --config configs/baseline.yaml --resume   # reprend depuis runs/<name>/ckpt/last.pth
    from train import train; metrics = train(load_config('configs/baseline.yaml'))   # depuis Python (tune.py)

Reproductibilité : même config + même machine => mêmes poids au bit près, y compris après --resume
(états RNG sauvegardés) et quelle que soit la fréquence d'évaluation (l'évaluation a son propre RNG).
"""
import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter

from data import train_dataset, train_loader
from evaluate import evaluate, evaluate_prof, format_metrics, generate, save_grid, to_unit_images
from model import Generator, Discriminator
from preprocessing import add_instance_noise, noise_sigma, training_set
from utils import (D_train, G_train, environment, get_device, git_commit, load_ckpt, load_config, make_deterministic,
                   save_ckpt, set_seed, unwrap, ema_update)

MNIST_DIM, Z_DIM = 784, 100


def fmt(seconds):
    """Durée lisible : 1h02m03s ou 2m03s."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'


def create_run_dir(cfg, resume, device):
    """Crée runs/<name>/ (jamais écrasé sans --resume) et y écrit config + commit + environnement."""
    run_dir = os.path.join(cfg['output']['dir'], cfg['name'])
    if os.path.exists(run_dir) and not resume:
        raise SystemExit(f"{run_dir} existe déjà : change 'name' dans le YAML ou relance avec --resume.")
    os.makedirs(os.path.join(run_dir, 'ckpt'), exist_ok=True)
    if not resume:
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump({**cfg, 'git_commit': git_commit(), 'environment': environment(device)}, f, sort_keys=False)
    return run_dir


def build_models(cfg, device):
    """G (architecture imposée), D et leurs optimiseurs Adam."""
    G = Generator(g_output_dim=MNIST_DIM).to(device)
    D = Discriminator(MNIST_DIM).to(device)
    if torch.cuda.device_count() > 1:
        G, D = nn.DataParallel(G), nn.DataParallel(D)
    opt = cfg['optimizer']
    G_opt = optim.Adam(G.parameters(), lr=opt['lr'], betas=tuple(opt['betas']))
    D_opt = optim.Adam(D.parameters(), lr=opt['lr'], betas=tuple(opt['betas']))
    return G, D, G_opt, D_opt


def loss_params(cfg):
    """(perte, gamma R1, gamma R2) de la section loss du YAML ; sans cette section, la BCE de la baseline."""
    loss = cfg.get('loss', {})
    return loss.get('name', 'bce'), loss.get('r1_gamma', 0.), loss.get('r2_gamma', 0.)


def train_one_epoch(loader, G, D, G_opt, D_opt, criterion, device, loss, G_ema=None, ema_decay=0.):
    """Une époque (un pas de D puis un pas de G par batch, puis la mise à jour de G_ema s'il existe) ;
    renvoie les moyennes des pertes et scores."""
    stats = np.zeros(4)
    for x, _ in loader:
        x = x.view(-1, MNIST_DIM).to(device)
        d_loss, d_real, d_fake = D_train(x, G, D, D_opt, criterion, device, *loss)
        g_loss = G_train(x, G, D, G_opt, criterion, device, loss[0])
        if G_ema is not None:
            ema_update(G_ema, G, ema_decay)
        stats += (d_loss, g_loss, d_real, d_fake)
    stats /= len(loader)
    return dict(loss_D=stats[0], loss_G=stats[1], D_real=stats[2], D_fake=stats[3])


def evaluate_epoch(images, device, ev):
    """Métriques sur 'val' : petit CNN, plus Inception et VGG (suffixe _prof) si la config contient eval.prof: true."""
    metrics = evaluate(images, device, ev['k'], reference='val')
    if ev.get('prof'):
        prof = evaluate_prof(images, device, reference='val')
        metrics.update(fid_prof=float(prof['fid']), precision_prof=float(prof['precision']), recall_prof=float(prof['recall']))
    return metrics


def log_epoch(writer, run_dir, epoch, metrics, G, fixed_z):
    """TensorBoard (courbes, grille à z fixe, métriques) + une ligne JSON dans runs/<name>/metrics.jsonl."""
    for key in ('loss_D', 'loss_G', 'D_real', 'D_fake', 'fid', 'precision', 'recall', 'fid_prof', 'precision_prof', 'recall_prof'):
        if key in metrics:
            writer.add_scalar(key, metrics[key], epoch)
    with torch.no_grad():
        writer.add_image('samples', torchvision.utils.make_grid(to_unit_images(G(fixed_z)), nrow=8), epoch)
    with open(os.path.join(run_dir, 'metrics.jsonl'), 'a') as f:
        f.write(json.dumps({'epoch': epoch, **metrics}) + '\n')


def train(cfg, resume=False, trial=None):
    """Entraîne le GAN décrit par cfg et renvoie les métriques de la dernière époque.

    Fichiers écrits dans runs/<name>/ :
        config.yaml, metrics.jsonl          configuration exacte et une ligne par époque (versionnés)
        G.pth, D.pth                        dernière époque (versionnés)
        G_best.pth, D_best.pth              meilleure époque selon le FID sur 'val', Inception si eval.prof (versionnés)
        G_raw.pth, G_best_raw.pth           avec ema.decay : G brut ; G.pth et G_best.pth sont alors le G moyenné
        ckpt/last.pth, ckpt/epoch_XXX.pth   état complet pour --resume (non versionnés)
        grid.png, events.out.tfevents.*     grille finale, TensorBoard
    """
    device = get_device()
    set_seed(cfg['seed'])
    make_deterministic()
    run_dir = create_run_dir(cfg, resume, device)
    dataset = training_set(train_dataset(), cfg, device)   # split 'train', voir data.py
    loader = train_loader(cfg['train']['batch_size'], cfg['seed'], dataset)
    G, D, G_opt, D_opt = build_models(cfg, device)
    D_input = add_instance_noise(D, cfg)   # D lui-même sans data.instance_noise
    criterion = nn.BCELoss()
    # EMA facultative (section ema du YAML) : G_ema sert à l'évaluation, aux grilles, à G.pth et G_best.pth.
    # Sans cette section, rien ne change (aucun tirage aléatoire en plus).
    ema_decay = cfg.get('ema', {}).get('decay', 0.)
    G_ema = copy.deepcopy(unwrap(G)).requires_grad_(False) if ema_decay else None
    G_out = G_ema if G_ema is not None else unwrap(G)

    start, best_fid = 1, float('inf')
    last_path = os.path.join(run_dir, 'ckpt', 'last.pth')
    if resume and os.path.exists(last_path):
        ckpt = load_ckpt(last_path, G, D, G_opt, D_opt, device)
        loader.generator.set_state(ckpt['loader_rng'])
        start, best_fid = ckpt['epoch'] + 1, ckpt['best_fid']
        if G_ema is not None:
            G_ema.load_state_dict(ckpt.get('G_ema', ckpt['G']))
        print(f'Reprise à l\'époque {start}')

    writer = SummaryWriter(run_dir)
    fixed_z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(0)).to(device)
    n_epochs, save_every, ev = cfg['train']['epochs'], cfg['output']['save_every'], cfg['eval']
    print(f'Run {run_dir} | device {device.type} | {len(loader.dataset)} images | époques {start}->{n_epochs} | '
          f'loss {loss_params(cfg)[0]} | EMA {ema_decay or "non"}')

    t0, metrics = time.time(), {}
    for epoch in range(start, n_epochs + 1):
        t_ep = time.time()
        if D_input is not D:
            D_input.sigma = noise_sigma(cfg, epoch)
        metrics = train_one_epoch(loader, G, D_input, G_opt, D_opt, criterion, device, loss_params(cfg),
                                  G_ema, ema_decay)

        # Évaluation sur 'val' tous les eval.every (et à la dernière époque). fork_rng restaure l'état du RNG
        # global à la sortie : évaluer (ou pas) ne change jamais la suite de l'entraînement.
        evaluated = ev['every'] and (epoch % ev['every'] == 0 or epoch == n_epochs)
        if evaluated:
            with torch.random.fork_rng():
                images = generate(G_out, ev['n_samples'], device)
                metrics.update(evaluate_epoch(images, device, ev))
            score = metrics['fid_prof' if ev.get('prof') else 'fid']
            if score < best_fid:  # la dernière époque n'est pas forcément la meilleure
                best_fid = score
                torch.save(G_out.state_dict(), os.path.join(run_dir, 'G_best.pth'))
                torch.save(unwrap(D).state_dict(), os.path.join(run_dir, 'D_best.pth'))
                if G_ema is not None:
                    torch.save(unwrap(G).state_dict(), os.path.join(run_dir, 'G_best_raw.pth'))
            if trial is not None:  # Optuna : rapport intermédiaire + abandon des essais peu prometteurs
                import optuna
                trial.report(metrics['fid'], epoch)
                if trial.should_prune():
                    writer.close()
                    raise optuna.TrialPruned()
        log_epoch(writer, run_dir, epoch, metrics, G_out, fixed_z)

        # Checkpoints : last.pth (+ RNG) et G.pth / D.pth à chaque époque, epoch_XXX.pth tous les save_every
        state = dict(epoch=epoch, seed=cfg['seed'], best_fid=best_fid, loader_rng=loader.generator.get_state())
        if G_ema is not None:
            state['G_ema'] = G_ema.state_dict()
        save_ckpt(last_path, G, D, G_opt, D_opt, **state)
        torch.save(G_out.state_dict(), os.path.join(run_dir, 'G.pth'))
        torch.save(unwrap(D).state_dict(), os.path.join(run_dir, 'D.pth'))
        if G_ema is not None:
            torch.save(unwrap(G).state_dict(), os.path.join(run_dir, 'G_raw.pth'))
        if epoch % save_every == 0 or epoch == n_epochs:
            save_ckpt(os.path.join(run_dir, 'ckpt', f'epoch_{epoch:03d}.pth'), G, D, G_opt, D_opt, **state)

        # Progression
        elapsed = time.time() - t0
        per_epoch = elapsed / (epoch - start + 1)
        scores = f" | FID={metrics['fid']:.2f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}" if evaluated else ''
        print(f"[{epoch:03d}/{n_epochs}] D={metrics['loss_D']:.3f} G={metrics['loss_G']:.3f} "
              f"D(x)={metrics['D_real']:.2f} D(G(z))={metrics['D_fake']:.2f}{scores} | {time.time() - t_ep:.1f}s/ep | "
              f'écoulé {fmt(elapsed)} | restant ~{fmt(per_epoch * (n_epochs - epoch))}', flush=True)

    writer.close()
    if 'fid' in metrics:  # grille 10x10 de la dernière évaluation
        save_grid(images, os.path.join(run_dir, 'grid.png'))
        print(format_metrics(metrics))
    print(f'Entraînement terminé en {fmt(time.time() - t0)}. Générateur : {run_dir}/G.pth (meilleur : G_best.pth)')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument('--config', default='configs/baseline.yaml', help='Fichier YAML de configuration.')
    parser.add_argument('--resume', action='store_true', help='Reprendre depuis <run>/ckpt/last.pth.')
    parser.add_argument('--seed', type=int, help='Remplace la seed du YAML et ajoute _s<seed> au nom du run.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.update(seed=args.seed, name=f"{cfg['name']}_s{args.seed}")
    train(cfg, resume=args.resume)
