"""Tests de sampling.py (CPU, modèles non entraînés : on teste la mécanique, pas la qualité).

Lancer :  python -m pytest tests -q
"""
import pytest
import torch

from model import Discriminator, Generator
from sampling import SAMPLERS, sample


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(0)
    return Generator(784).eval(), Discriminator(784).eval()


def test_plain_is_registered():
    assert 'plain' in SAMPLERS


def test_exact_number_of_images_in_range(models):
    x, stats = sample(*models, 300, {'method': 'plain'}, seed=0, batch_size=128)
    assert x.shape == (300, 784) and x.min() >= -1 and x.max() <= 1
    assert stats['n_proposals'] == 300 and stats['acceptance_rate'] == 1.0


def test_same_seed_same_images(models):
    x1, _ = sample(*models, 64, {'method': 'plain'}, seed=7)
    x2, _ = sample(*models, 64, {'method': 'plain'}, seed=7)
    assert torch.equal(x1, x2)
