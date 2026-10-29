import torch
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



def D_train(x, G, D, D_optimizer, criterion, device):
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


def G_train(x, G, D, G_optimizer, criterion, device):
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
    ckpt_path = os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
