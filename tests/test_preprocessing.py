"""Tests des options de données : python -m pytest tests/test_preprocessing.py"""
import torch

from model import Discriminator
from preprocessing import InstanceNoise, gaussian_score, noise_sigma, select_instances, training_set


def toy(n=300, dim=4):
    torch.manual_seed(0)
    return torch.arange(n) % 3, torch.randn(n, dim)


def test_gaussian_score_ranks_outlier_last():
    f = torch.randn(500, 4)
    f[7] = 20
    assert gaussian_score(f).argmin() == 7


def test_per_class_keeps_the_class_mix():
    y, f = toy()
    idx = select_instances(y, f, 0.5)
    assert len(idx) == len(idx.unique()) == 150
    assert torch.bincount(y[idx]).tolist() == [50, 50, 50]


def test_global_can_empty_a_class():
    y, f = toy()
    f[y == 2] += torch.tensor([50., 0, 0, 0]) * torch.randn(100, 1)  # classe très dispersée
    assert torch.bincount(y[select_instances(y, f, 0.5, per_class=False)], minlength=3)[2] < 30


def test_selection_off_returns_the_same_dataset():
    ds = object()
    for data in ({}, {'selection': {'enabled': False, 'keep': 0.5}}):
        assert training_set(ds, {'data': data}, 'cpu') is ds




def test_sigma_schedule():
    on = lambda **kw: {'data': {'instance_noise': {'enabled': True, 'sigma': 0.2, **kw}}}
    assert noise_sigma({'data': {}}, 1) == 0
    assert noise_sigma({'data': {'instance_noise': {'enabled': False, 'sigma': 0.2}}}, 1) == 0
    assert noise_sigma(on(anneal_epochs=0), 150) == 0.2
    assert noise_sigma(on(anneal_epochs=100), 1) == 0.2
    assert abs(noise_sigma(on(anneal_epochs=100), 51) - 0.1) < 1e-9
    assert noise_sigma(on(anneal_epochs=100), 101) == 0


def test_zero_sigma_is_the_identity_and_draws_nothing():
    D, x = Discriminator(784), torch.zeros(2, 784)
    wrapped = InstanceNoise(D)
    state = torch.get_rng_state()
    assert torch.equal(wrapped(x), D(x))
    assert torch.equal(state, torch.get_rng_state())


def test_noise_has_the_requested_std():
    seen = {}
    wrapped = InstanceNoise(lambda x: seen.update(x=x))
    wrapped.sigma = 0.3
    wrapped(torch.zeros(100_000, 1))
    assert abs(seen['x'].std().item() - 0.3) < 0.01
