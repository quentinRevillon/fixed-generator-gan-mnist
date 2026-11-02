import torch
import torch.nn.functional as F
import os
import random
import subprocess
import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')



def D_logits(D, x):
    # sortie de D avant la sigmoïde finale (model.py reste inchangé)
    D = unwrap(D)
    for fc in (D.fc1, D.fc2, D.fc3):
        x = F.leaky_relu(fc(x), 0.2)
    return D.fc4(x)


def zero_centered_gp(samples, logits):
    # ||grad_x D(x)||^2 moyenné sur le batch : R1 sur les vraies images, R2 sur les fausses
    grad, = torch.autograd.grad(outputs=logits.sum(), inputs=samples, create_graph=True)
    return grad.pow(2).sum(dim=1).mean()


def D_train(x, G, D, D_optimizer, criterion, device, loss='bce', r1_gamma=0., r2_gamma=0.):
    if loss == 'rpgan':
        return D_train_rpgan(x, G, D, D_optimizer, device, r1_gamma, r2_gamma)

    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item(), D_real_score.mean().item(), D_fake_score.mean().item()


def D_train_rpgan(x, G, D, D_optimizer, device, r1_gamma, r2_gamma):
    #=========Train the discriminator: RpGAN + R1 + R2 (R3GAN, Huang et al. 2024)=========#
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


def G_train(x, G, D, G_optimizer, criterion, device, loss='bce'):
    if loss == 'rpgan':
        return G_train_rpgan(x, G, D, G_optimizer, device)

    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def G_train_rpgan(x, G, D, G_optimizer, device):
    #=======================Train the generator: RpGAN=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    fake_logits = D_logits(D, G(z))
    with torch.no_grad():
        real_logits = D_logits(D, x.to(device))

    # paire relativiste : G veut D(faux) > D(réel)
    G_loss = F.softplus(real_logits - fake_logits).mean()
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_ckpt(path, G, D, G_opt, D_opt, epoch, seed):
    torch.save({'G': unwrap(G).state_dict(), 'D': unwrap(D).state_dict(),
                'G_opt': G_opt.state_dict(), 'D_opt': D_opt.state_dict(),
                'epoch': epoch, 'seed': seed}, path)


def load_ckpt(path, G, D, G_opt, D_opt, device):
    ckpt = torch.load(path, map_location=device)
    unwrap(G).load_state_dict(ckpt['G'])
    unwrap(D).load_state_dict(ckpt['D'])
    G_opt.load_state_dict(ckpt['G_opt'])
    D_opt.load_state_dict(ckpt['D_opt'])
    return ckpt['epoch']


def load_model(G, folder, device):
    # folder peut aussi être directement un fichier .pth
    ckpt_path = folder if folder.endswith('.pth') else os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
