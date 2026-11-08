"""Diagnostics du DRS (Azadi et al. 2019) sur CPU : pourquoi il aide la baseline et pas RpGAN + R1 + EMA.

Lancer depuis le worktree (lecture seule, aucun fichier écrit dedans) :
    cd wt-comparaison
    DATA=<scratchpad>/data python -B <scratchpad>/scripts/drs/diag_drs.py

Pour chaque couple (G, D) :
    scores    : d(x) sur MNIST test et sur G(z) ; AUC ; log E_G[e^d] ; recalibration de Platt (a, b)
    metrics   : CNN FID / précision / rappel contre test, sans tri, DRS p50/p80/p95, DRS p80 avec D recalibré
                (Platt sur le logit, ou régression logistique sur les features de fc3), OBRS K=3 ;
                plus le DRS p80 ramené aux proportions de chiffres sans tri (isole l'effet « mélange des chiffres »)
    classes   : proportion de chaque chiffre prédit et probabilité d'acceptation moyenne par chiffre (p80)
Sorties CSV dans <scratchpad>/results/drs/.
"""
import copy
import csv
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
WT = r'C:\Users\quent\Education\IASD\datalab\datalab 2\wt-comparaison'
SCR = r'C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad'
sys.path.insert(0, WT)
os.chdir(WT)
torch.set_num_threads(2)

from data import mnist_split  # noqa: E402
from evaluate import evaluate, fid, load_classifier, predict, to_unit_images  # noqa: E402
from model import Discriminator, Generator  # noqa: E402
from sampling import batched_logits, drs_f_hat, gaussian_latents, generate_from_latents, sample  # noqa: E402
from utils import load_model  # noqa: E402

DEV = torch.device('cpu')
M = os.path.join(SCR, 'models', 'drs')
OUT = os.path.join(SCR, 'results', 'drs')
MAIN = r'C:\Users\quent\Education\IASD\datalab\datalab 2\masteriasd-datalab-project-a2-group-3'
MODELS = {  # nom -> (G, D)
    'bl200_prof': (f'{WT}/runs/baseline_200ep_prof/G_best.pth', f'{WT}/runs/baseline_200ep_prof/D_best.pth'),
    **{f'bl_s{s}': (f'{M}/bl_s{s}_G.pth', f'{M}/bl_s{s}_D.pth') for s in range(3)},
    **{f'rp_s{s}_ema': (f'{M}/rp_s{s}_G.pth', f'{M}/rp_s{s}_D.pth') for s in range(3)},
    **{f'rp_s{s}_raw': (f'{M}/rp_s{s}_G_raw.pth', f'{M}/rp_s{s}_D.pth') for s in range(3)},
}
SEEDS = tuple(int(s) for s in os.getenv('SEEDS', '0 1').split())
N = 10_000


def load(g_path, d_path):
    """(G, D) en mode évaluation sur CPU."""
    return (load_model(Generator(784), g_path, DEV).eval(), load_model(Discriminator(784), d_path, DEV).eval())


def fake(G, n, seed):
    """n images G(z) (n, 784) dans [-1, 1], latents tirés avec cette seed."""
    return generate_from_latents(G, gaussian_latents(n, torch.Generator().manual_seed(seed)), N)


def real(split):
    """Images réelles du split, au format de sortie de G : (n, 784) dans [-1, 1]."""
    return mnist_split(split)[0].view(-1, 784) * 2 - 1


@torch.no_grad()
def fc3_features(D, x):
    """Sortie de fc3 de D (après leaky ReLU), l'entrée de la dernière couche fc4."""
    h = x
    for fc in (D.fc1, D.fc2, D.fc3):
        h = torch.nn.functional.leaky_relu(fc(h), 0.2)
    return h


def with_last_layer(D, w, b):
    """Copie de D dont la couche fc4 vaut w·h + b : le logit recalibré, utilisable tel quel par sampling.py."""
    D2 = copy.deepcopy(D)
    D2.fc4.weight.data = torch.as_tensor(w, dtype=torch.float32).view(1, -1)
    D2.fc4.bias.data = torch.as_tensor([b], dtype=torch.float32).view(1)
    return D2


def platt(D, x_real, x_fake):
    """Régression logistique 1D sur le logit : d' = a d + b. Renvoie (a, b, D recalibré)."""
    d = torch.cat([batched_logits(D, x_real, N), batched_logits(D, x_fake, N)]).numpy()[:, None]
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(d, y)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    return a, b, with_last_layer(D, a * D.fc4.weight.data.view(-1), a * D.fc4.bias.item() + b)


def feature_lr(D, x_real, x_fake):
    """Régression logistique sur les 256 features de fc3 (réels vs G) : nouvelle dernière couche de D."""
    h = torch.cat([fc3_features(D, x_real), fc3_features(D, x_fake)]).numpy()
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))]
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(h, y)
    return with_last_layer(D, lr.coef_[0], lr.intercept_[0])


def auc(D, x_real, x_fake):
    """AUC de D pour séparer réels et faux (0,5 = hasard)."""
    d = torch.cat([batched_logits(D, x_real, N), batched_logits(D, x_fake, N)]).numpy()
    return roc_auc_score(np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))], d)


def log_mean_exp(d):
    """log E[e^d] ; vaut 0 si d est exactement log p_data / p_G et d est évalué sur G."""
    return (torch.logsumexp(d.double(), 0) - np.log(len(d))).item()


def cnn_metrics(x):
    """CNN FID, précision, rappel contre MNIST test et classes prédites, pour x (n, 784) dans [-1, 1]."""
    m = evaluate(to_unit_images(x), DEV, k=3, reference='test')
    return m['fid'], m['precision'], m['recall'], m['classes']


def match_classes(x, target_counts, clf):
    """Sous-échantillon de x (sans remise) qui a exactement target_counts images de chaque chiffre prédit."""
    preds = predict(clf, to_unit_images(x), DEV)[1]
    idx = [torch.nonzero(preds == c).view(-1)[:n] for c, n in enumerate(target_counts)]
    assert all(len(i) == n for i, n in zip(idx, target_counts)), 'pas assez d\'images pour un chiffre'
    return x[torch.cat(idx)]


def score_row(name, G, D, x_test, x_val):
    """Statistiques des scores de D, et recalibrations ajustées sur val vs G (seed 100)."""
    x_g = fake(G, N, 100)
    d_real, d_fake = batched_logits(D, x_test, N), batched_logits(D, fake(G, N, 200), N)
    a, b, D_platt = platt(D, x_val, x_g)
    D_feat = feature_lr(D, x_val, x_g)
    x_g_test = fake(G, N, 200)
    hist = np.histogram(np.r_[d_real.numpy(), d_fake.numpy()], bins=60)[1]
    with open(f'{OUT}/hist_{name}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bin_left', 'real_count', 'fake_count'])
        for e, r, g in zip(hist[:-1], np.histogram(d_real, hist)[0], np.histogram(d_fake, hist)[0]):
            w.writerow([round(float(e), 4), r, g])
    row = dict(model=name, d_real_mean=d_real.mean().item(), d_real_std=d_real.std().item(),
               d_fake_mean=d_fake.mean().item(), d_fake_std=d_fake.std().item(), d_fake_max=d_fake.max().item(),
               auc_raw=auc(D, x_test, x_g_test), auc_feat=auc(D_feat, x_test, x_g_test),
               logmeanexp_d_on_G=log_mean_exp(d_fake), logmeanexp_minus_d_on_real=log_mean_exp(-d_real),
               platt_a=a, platt_b=b, logmeanexp_platt_on_G=log_mean_exp(a * d_fake + b))
    return row, D_platt, D_feat


def class_acceptance(G, D, clf, percentile=80):
    """Proportion de chaque chiffre prédit dans 10 000 propositions et acceptation DRS moyenne par chiffre."""
    x = fake(G, N, 300)
    d = batched_logits(D, x, N)
    f = drs_f_hat(d, d.max().item(), 0.0, 1e-6)
    acc = torch.sigmoid(f - torch.quantile(f, percentile / 100))
    preds = predict(clf, to_unit_images(x), DEV)[1]
    return [(c, (preds == c).float().mean().item(), acc[preds == c].mean().item()) for c in range(10)]


def main():
    os.makedirs(OUT, exist_ok=True)
    clf = load_classifier(DEV)
    x_test, x_val = real('test'), real('val')
    test_counts = torch.bincount(mnist_split('test')[1], minlength=10).tolist()
    score_rows, metric_rows, class_rows = [], [], []
    only = sys.argv[1:]
    for name, (g_path, d_path) in MODELS.items():
        if only and name not in only:
            continue
        G, D = load(g_path, d_path)
        srow, D_platt, D_feat = score_row(name, G, D, x_test, x_val)
        score_rows.append(srow)
        print(srow, flush=True)
        for c, share, acc in class_acceptance(G, D, clf):
            class_rows.append(dict(model=name, digit=c, share_plain=share, share_test=test_counts[c] / sum(test_counts),
                                   accept_p80=acc))
        variants = [('plain', D, {'method': 'plain'}),
                    ('drs_p50', D, {'method': 'drs_azadi', 'percentile': 50}),
                    ('drs_p80', D, {'method': 'drs_azadi', 'percentile': 80}),
                    ('drs_p95', D, {'method': 'drs_azadi', 'percentile': 95}),
                    ('drs_p80_platt', D_platt, {'method': 'drs_azadi', 'percentile': 80}),
                    ('drs_p80_featlr', D_feat, {'method': 'drs_azadi', 'percentile': 80}),
                    ('obrs_K3', D, {'method': 'obrs', 'K': 3}),
                    ('obrs_K3_featlr', D_feat, {'method': 'obrs', 'K': 3})]
        for seed in SEEDS:
            plain_counts = None
            for label, Dv, cfg in variants:
                x, stats = sample(G, Dv, N, cfg, seed=seed, batch_size=N)
                f_, p_, r_, cls = cnn_metrics(x)
                plain_counts = cls if label == 'plain' else plain_counts
                metric_rows.append(dict(model=name, variant=label, seed=seed, fid_cnn=f_, precision_cnn=p_, recall_cnn=r_,
                                        acceptance=stats['acceptance_rate'], classes='-'.join(map(str, cls)),
                                        tv_to_test=0.5 * sum(abs(c / N - t / sum(test_counts)) for c, t in zip(cls, test_counts))))
                print(metric_rows[-1], flush=True)
            # DRS p80 ramené aux proportions de chiffres de « sans tri » : seul l'effet intra-chiffre reste
            x_big, _ = sample(G, D, 6 * N, {'method': 'drs_azadi', 'percentile': 80}, seed=seed, batch_size=N)
            for label, counts in (('drs_p80_plain_mix', plain_counts), ('drs_p80_test_mix', test_counts)):
                x = match_classes(x_big, counts, clf)
                if x is None:
                    continue
                f_, p_, r_, cls = cnn_metrics(x)
                metric_rows.append(dict(model=name, variant=label, seed=seed, fid_cnn=f_, precision_cnn=p_, recall_cnn=r_,
                                        acceptance=float('nan'), classes='-'.join(map(str, cls)),
                                        tv_to_test=0.5 * sum(abs(c / N - t / sum(test_counts)) for c, t in zip(cls, test_counts))))
                print(metric_rows[-1], flush=True)
            x_plain_big = sample(G, D, 6 * N, {'method': 'plain'}, seed=seed + 10, batch_size=N)[0]
            x = match_classes(x_plain_big, test_counts, clf)
            if x is None:
                continue
            f_, p_, r_, cls = cnn_metrics(x)
            metric_rows.append(dict(model=name, variant='plain_test_mix', seed=seed, fid_cnn=f_, precision_cnn=p_, recall_cnn=r_,
                                    acceptance=1.0, classes='-'.join(map(str, cls)), tv_to_test=0.0))
            print(metric_rows[-1], flush=True)
        suffix = '_' + '_'.join(only) if only else ''
        for fname, rows in (('scores.csv', score_rows), ('metrics_cnn.csv', metric_rows), ('classes.csv', class_rows)):
            with open(f'{OUT}/{fname[:-4]}{suffix}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


if __name__ == '__main__':
    main()
