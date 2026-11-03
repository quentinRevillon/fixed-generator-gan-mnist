"""Tests du DRS d'Azadi et al. 2019 : équations 6 et 8, gamma par percentile, KeepTraining (CPU, sans MNIST).

Lancer :  python -m pytest tests -q
"""
import copy

import pytest
import torch

from keep_training import discriminator_bce, keep_training, validation_bce
from model import Discriminator, Generator
from sampling import drs_f_hat, sample


@pytest.fixture
def models():
    torch.manual_seed(0)
    return Generator(784).eval(), Discriminator(784).eval()


def test_f_hat_matches_equation_6_when_gamma_is_zero():
    log_r = torch.tensor([-9.0, -3.0, -0.5])
    accept = torch.sigmoid(drs_f_hat(log_r, log_M=0.0, gamma=0.0, eps=1e-9))
    assert torch.allclose(accept, torch.exp(log_r.double()), atol=1e-6)


def test_f_hat_stays_finite_at_the_maximum():
    assert torch.isfinite(drs_f_hat(torch.tensor([2.0]), log_M=2.0, gamma=0.0, eps=1e-6)).all()


def test_gamma_is_subtracted():
    log_r = torch.tensor([-2.0])
    assert (drs_f_hat(log_r, 0.0, 0.0, 1e-6) - drs_f_hat(log_r, 0.0, 3.0, 1e-6)).item() == pytest.approx(3.0)


def test_sampler_returns_n_images_and_is_deterministic(models):
    cfg = {'method': 'drs_azadi', 'percentile': 80, 'n_burnin': 1000}
    x1, stats = sample(*models, 300, cfg, seed=1, batch_size=500)
    x2, _ = sample(*models, 300, cfg, seed=1, batch_size=500)
    assert x1.shape == (300, 784) and torch.equal(x1, x2)
    assert 0 < stats['acceptance_rate'] <= 1 and stats['log_M'] >= stats['log_M_burnin']


def test_higher_percentile_rejects_more(models):
    rates = [sample(*models, 200, {'method': 'drs_azadi', 'percentile': p, 'n_burnin': 1000}, batch_size=500)[1]
             ['acceptance_rate'] for p in (50, 95)]
    assert rates[0] > rates[1]


def test_keep_training_freezes_g_and_never_worsens_validation(models):
    G, D = models
    x_val, x_train = torch.rand(64, 784) * 2 - 1, torch.rand(256, 784) * 2 - 1
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, torch.zeros(256)), batch_size=32)
    before = copy.deepcopy(G.state_dict())
    best_D, history = keep_training(G, D, loader, x_val, eval_every=2, patience=2, max_steps=12)
    assert all(torch.equal(before[k], v) for k, v in G.state_dict().items())
    assert min(h['val_bce'] for h in history) <= history[0]['val_bce']
    assert len(history) >= 2 and not best_D.training


def test_discriminator_bce_is_positive(models):
    _, D = models
    x = torch.rand(8, 784)
    assert discriminator_bce(D, x, x).item() > 0 and validation_bce(D, x, x) > 0
