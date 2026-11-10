"""Figure of generated samples: the same 36 latent vectors z through four generators (visual check of the platform).

Usage, from rapport/report/:
    python figscripts/samples.py      # -> figures/combination_samples.pdf (+ .png)

Weights: runs/<run>/G.pth (G_EMA when the run uses EMA) and G_raw.pth (raw G) of the main repository
(masteriasd-datalab-project-a2-group-3, not versioned). Generator: model.py of the repository.
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision

sys.path.insert(0, 'figscripts')
sys.path.insert(0, os.path.join('..', '..'))  # model.py of the repository (worktree root)
from model import Generator  # noqa: E402
from style import FULL, apply, save  # noqa: E402

RUNS = os.path.join('..', '..', '..', 'masteriasd-datalab-project-a2-group-3', 'runs')
PANELS = [('(a) Baseline', 'cmploc_baseline_s1/G.pth'),
          ('(b) New loss', 'cmploc_rpgan_r1_ema_s1/G_raw.pth'),
          ('(c) New loss + EMA (submitted)', 'cmploc_rpgan_r1_ema_s1/G.pth'),
          ('(d) A collapsed baseline run', 'ema100_baseline_s2/G.pth')]


def load_generator(path):
    """Loads a Generator from a .pth state dict (strips DataParallel 'module.' prefixes)."""
    state = torch.load(path, map_location='cpu')
    G = Generator(784)
    G.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    return G.eval()


@torch.no_grad()
def grid(G, z, nrow=6):
    """Image grid (H, W) in [0, 1] of G(z), white 1-pixel separators."""
    x = (G(z).view(-1, 1, 28, 28) + 1) / 2
    return torchvision.utils.make_grid(x, nrow=nrow, padding=1, pad_value=1)[0].numpy()


def main():
    """Builds figures/combination_samples.pdf: one panel per generator, same z (seed 0), full page width."""
    apply()
    z = torch.randn(36, 100, generator=torch.Generator().manual_seed(0))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(FULL, 1.9))
    for ax, (title, rel) in zip(axes, PANELS):
        ax.imshow(grid(load_generator(os.path.join(RUNS, rel)), z), cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_title(title, fontsize=8.5)
        ax.axis('off')
    save(fig, 'combination_samples', FULL)


if __name__ == '__main__':
    main()
