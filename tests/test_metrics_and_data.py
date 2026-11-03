"""Tests des métriques (FID, precision / recall k-NN) et du split (sans téléchargement de MNIST).

Lancer :  python -m pytest tests -q
"""
import pytest
import torch

from data import VAL_SIZE, split_indices
from evaluate import coverage, fid, knn_radii, to_unit_images


def test_split_is_disjoint_complete_and_fixed():
    train, val = split_indices()
    assert len(train) == 60_000 - VAL_SIZE and len(val) == VAL_SIZE
    assert len(set(train.tolist()) | set(val.tolist())) == 60_000
    assert torch.equal(split_indices()[1], val)


def test_fid_identical_sets_is_zero():
    f = torch.randn(2000, 16)
    assert fid(f, f) == pytest.approx(0, abs=1e-6)


def test_fid_detects_mean_shift():
    f = torch.randn(2000, 16)
    assert fid(f, f + 1) == pytest.approx(16, rel=0.05)     # ||mu_r - mu_g||^2 = 16 x 1^2


def test_precision_recall_identical_sets_are_one():
    f = torch.randn(1000, 8)
    assert coverage(f, f, knn_radii(f, k=3)) == 1.0


def test_precision_drops_for_far_away_fakes():
    real, fake = torch.randn(1000, 8), torch.randn(1000, 8) + 10
    assert coverage(fake, real, knn_radii(real, k=3)) == 0.0


def test_to_unit_images_inverts_normalisation():
    x = torch.tensor([[-1.0, 0.0, 1.0] + [0.0] * 781])
    img = to_unit_images(x)
    assert img.shape == (1, 1, 28, 28) and img.flatten()[:3].tolist() == [0.0, 0.5, 1.0]
