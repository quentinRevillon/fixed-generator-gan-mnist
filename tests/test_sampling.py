"""Tests de sampling.py (CPU, modèles non entraînés : on teste la mécanique, pas la qualité).

Lancer :  python -m pytest tests -q
"""
import math

import pytest
import torch

from sampling import (SAMPLERS, drs_accept_prob, drs_gamma, obrs_accept_prob, obrs_threshold, sample,
                      truncated_latents)
from model import Discriminator, Generator

CONFIGS = [{'method': 'plain'}, {'method': 'truncation', 'psi': 1.0},
           {'method': 'obrs', 'K': 3, 'n_calib': 5000}, {'method': 'drs', 'K': 3, 'n_calib': 5000},
           {'method': 'drs_azadi', 'percentile': 80, 'n_burnin': 2000}]


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(0)
    return Generator(784).eval(), Discriminator(784).eval()


def test_every_method_is_registered():
    assert {c['method'] for c in CONFIGS} == set(SAMPLERS)


@pytest.mark.parametrize('cfg', CONFIGS, ids=lambda c: c['method'])
def test_exact_number_of_images_in_range(models, cfg):
    x, stats = sample(*models, 1234, cfg, seed=0, batch_size=1000)
    assert x.shape == (1234, 784)
    assert x.min() >= -1 and x.max() <= 1
    assert 0 < stats['acceptance_rate'] <= 1 and stats['n_proposals'] >= 1234


@pytest.mark.parametrize('cfg', CONFIGS, ids=lambda c: c['method'])
def test_same_seed_same_images(models, cfg):
    x1, _ = sample(*models, 500, cfg, seed=7, batch_size=1000)
    x2, _ = sample(*models, 500, cfg, seed=7, batch_size=1000)
    x3, _ = sample(*models, 500, cfg, seed=8, batch_size=1000)
    assert torch.equal(x1, x2) and not torch.equal(x1, x3)


@pytest.mark.parametrize('K', [1.5, 2, 5, 10])
def test_obrs_threshold_gives_budget(K):
    log_r = torch.randn(20_000) * 2
    tau = obrs_threshold(log_r, K)
    assert obrs_accept_prob(log_r, tau).mean().item() == pytest.approx(1 / K, abs=1e-4)


def test_obrs_large_budget_is_exact_rejection_sampling():
    log_r = torch.randn(1000)
    tau = obrs_threshold(log_r, K=1e6)          # K > M : c_K = 1, tau = log M
    assert tau == pytest.approx(log_r.max().item())


def test_obrs_budget_one_accepts_everything():
    log_r = torch.randn(1000)
    assert obrs_accept_prob(log_r, obrs_threshold(log_r, K=1)).min().item() == pytest.approx(1.0)


@pytest.mark.parametrize('K', [2, 5, 10])
def test_drs_gamma_gives_budget(K):
    log_r = torch.randn(20_000) * 2
    log_M = log_r.max().item()
    gamma = drs_gamma(log_r, log_M, K)
    assert drs_accept_prob(log_r, log_M, gamma).mean().item() == pytest.approx(1 / K, abs=1e-4)


def test_drs_gamma_zero_is_exact_rejection_sampling():
    log_r = torch.randn(1000)
    log_M = log_r.max().item()
    a = drs_accept_prob(log_r, log_M, gamma=0.0)
    assert torch.allclose(a, torch.exp(log_r - log_M), atol=1e-6)


def test_truncated_latents_bounds_and_variance():
    psi = 1.0
    z = truncated_latents(100_000, psi, torch.Generator().manual_seed(0))
    assert z.abs().max() <= psi
    phi = math.exp(-psi ** 2 / 2) / math.sqrt(2 * math.pi)
    var = 1 - 2 * psi * phi / (2 * (0.5 * (1 + math.erf(psi / math.sqrt(2)))) - 1)
    assert z.var().item() == pytest.approx(var, rel=0.01)
