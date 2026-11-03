"""Tests de la mesure du prof (Inception, VGG), sans téléchargement : les réseaux sont remplacés par des features simples.

Lancer :  python -m pytest tests -q
"""
import pytest
import torch

import evaluate as ev
import train


@pytest.fixture
def fake_networks(monkeypatch):
    """Remplace Inception, VGG, le classifieur et MNIST ; renvoie les images réelles factices."""
    real = torch.rand(40, 1, 28, 28)
    monkeypatch.setattr(ev, 'mnist_split', lambda reference: (real, None))
    monkeypatch.setattr(ev, 'inception_features', lambda x, device: x.flatten(1)[:, :8])
    monkeypatch.setattr(ev, 'vgg_features', lambda x, device: x.flatten(1)[:, :8])
    monkeypatch.setattr(ev, 'load_classifier', lambda device: None)
    monkeypatch.setattr(ev, 'predict', lambda clf, x, device: (None, torch.zeros(len(x), dtype=torch.long)))
    monkeypatch.setattr(ev, '_prof_cache', {})
    return real


def test_identical_images_give_perfect_scores(fake_networks):
    m = ev.evaluate_prof(fake_networks, torch.device('cpu'), reference='val')
    assert m['fid'] == pytest.approx(0, abs=1e-5)
    assert m['precision'] == 1.0 and m['recall'] == 1.0
    assert m['classes'][0] == 40 and m['reference'] == 'val'


def test_precision_recall_use_at_most_9984_images(fake_networks, monkeypatch):
    seen = []
    monkeypatch.setattr(ev, 'vgg_features', lambda x, device: seen.append(len(x)) or x.flatten(1)[:, :8])
    monkeypatch.setattr(ev, 'PROF_PR_IMAGES', 10)
    ev.evaluate_prof(fake_networks, torch.device('cpu'))
    assert seen == [10, 10]


def test_vgg_preprocess_gives_imagenet_input():
    x = ev.vgg_preprocess(torch.ones(2, 1, 28, 28), torch.device('cpu'))
    assert x.shape == (2, 3, 224, 224)
    assert x[0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)


def test_metrics_write_to_separate_tables():
    assert ev.RESULTS_CSV['cnn'] != ev.RESULTS_CSV['prof']
    assert set(ev.EVALUATORS) == {'cnn', 'prof'}


def test_training_evaluation_adds_the_leaderboard_measure_on_request(monkeypatch):
    monkeypatch.setattr(train, 'evaluate', lambda images, device, k, reference: {'fid': 1.0})
    monkeypatch.setattr(train, 'evaluate_prof', lambda images, device, reference: {'fid': 2.0, 'precision': 0.5, 'recall': 0.25})
    assert train.evaluate_epoch(None, 'cpu', {'k': 3}) == {'fid': 1.0}
    assert train.evaluate_epoch(None, 'cpu', {'k': 3, 'prof': True}) == {
        'fid': 1.0, 'fid_prof': 2.0, 'precision_prof': 0.5, 'recall_prof': 0.25}
