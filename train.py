import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import Generator, Discriminator
from utils import (D_train, G_train, load_config, set_seed, git_commit, get_device,
                   save_ckpt, load_ckpt, unwrap)

MNIST_DIM, Z_DIM = 784, 100


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'


def train(cfg, resume=False, trial=None):
    """Entraîne le GAN décrit par cfg et renvoie les métriques de la dernière époque."""
    device = get_device()
    set_seed(cfg['seed'])

    # Dossier du run : jamais écrasé sans --resume
    run_dir = os.path.join(cfg['output']['dir'], cfg['name'])
    if os.path.exists(run_dir) and not resume:
        raise SystemExit(f"{run_dir} existe déjà : change 'name' dans le YAML ou relance avec --resume.")
    os.makedirs(os.path.join(run_dir, 'ckpt'), exist_ok=True)
    if not resume:
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump({**cfg, 'git_commit': git_commit()}, f, sort_keys=False)

    # Données : $DATA sur Juliet, sinon téléchargement dans data_path
    data_path = os.getenv('DATA', cfg['data']['path'])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download='DATA' not in os.environ)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['train']['num_workers'], pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(cfg['seed']))

    # Modèles et optimiseurs
    G = Generator(g_output_dim=MNIST_DIM).to(device)
    D = Discriminator(MNIST_DIM).to(device)
    if torch.cuda.device_count() > 1:
        G, D = nn.DataParallel(G), nn.DataParallel(D)
    criterion = nn.BCELoss()
    opt = cfg['optimizer']
    G_opt = optim.Adam(G.parameters(), lr=opt['lr'], betas=tuple(opt['betas']))
    D_opt = optim.Adam(D.parameters(), lr=opt['lr'], betas=tuple(opt['betas']))

    start = 1
    last_path = os.path.join(run_dir, 'ckpt', 'last.pth')
    if resume and os.path.exists(last_path):
        start = load_ckpt(last_path, G, D, G_opt, D_opt, device) + 1
        print(f'Reprise à l\'époque {start}')

    writer = SummaryWriter(run_dir)
    fixed_z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(0)).to(device)
    n_epochs, save_every = cfg['train']['epochs'], cfg['output']['save_every']
    print(f'Run {run_dir} | device {device.type} | {len(dataset)} images | époques {start}->{n_epochs}')

    t0, metrics = time.time(), {}
    for epoch in range(start, n_epochs + 1):
        t_ep, stats = time.time(), np.zeros(4)
        for x, _ in loader:
            x = x.view(-1, MNIST_DIM).to(device)
            d_loss, d_real, d_fake = D_train(x, G, D, D_opt, criterion, device)
            g_loss = G_train(x, G, D, G_opt, criterion, device)
            stats += (d_loss, g_loss, d_real, d_fake)
        stats /= len(loader)
        metrics = dict(epoch=epoch, loss_D=stats[0], loss_G=stats[1], D_real=stats[2], D_fake=stats[3])

        # TensorBoard : courbes + grille à z fixe (ramenée dans [0, 1])
        for tag, v in zip(['loss/D', 'loss/G', 'D(x)', 'D(G(z))'], stats):
            writer.add_scalar(tag, v, epoch)
        with torch.no_grad():
            grid = (G(fixed_z).view(-1, 1, 28, 28) + 1) / 2
        writer.add_image('samples', torchvision.utils.make_grid(grid, nrow=8), epoch)

        # Checkpoints : last.pth + G.pth à chaque époque, epoch_XXX.pth tous les save_every
        save_ckpt(last_path, G, D, G_opt, D_opt, epoch, cfg['seed'])
        torch.save(unwrap(G).state_dict(), os.path.join(run_dir, 'G.pth'))
        if epoch % save_every == 0 or epoch == n_epochs:
            save_ckpt(os.path.join(run_dir, 'ckpt', f'epoch_{epoch:03d}.pth'), G, D, G_opt, D_opt, epoch, cfg['seed'])

        # Progression
        elapsed = time.time() - t0
        per_epoch = elapsed / (epoch - start + 1)
        print(f'[{epoch:03d}/{n_epochs}] D={stats[0]:.3f} G={stats[1]:.3f} '
              f'D(x)={stats[2]:.2f} D(G(z))={stats[3]:.2f} | {time.time() - t_ep:.1f}s/ep | '
              f'écoulé {fmt(elapsed)} | restant ~{fmt(per_epoch * (n_epochs - epoch))}', flush=True)

    writer.close()
    print(f'Entraînement terminé en {fmt(time.time() - t0)}. Générateur : {run_dir}/G.pth')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument('--config', default='configs/baseline.yaml', help='Fichier YAML de configuration.')
    parser.add_argument('--resume', action='store_true', help='Reprendre depuis <run>/ckpt/last.pth.')
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)
