"""Tableaux et figures de rapport-donnée-illustré.md. À lancer depuis la racine : python figures/make_data_figures.py [--real | --per-class]

--real : mesures sur les vraies images seules (aucun GAN) ; --per-class : FID par chiffre (long) ;
sans option : bilan des runs de configs/data/.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import torch
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, '.')
from evaluate import evaluate, fid, generate, load_classifier, mnist, predict
from model import Generator
from preprocessing import embed, gaussian_score, select_instances
from utils import load_model, set_seed

OUT = 'figures/data'
DEVICE = torch.device('cpu')
BUDGET = 60000  # mises à jour : comparaison à budget égal, sans choisir le meilleur point sur le test
SUBSETS = {'sel_95': (0.95, True, 'classifier'), 'sel_90': (0.9, True, 'classifier'),
           'sel_75': (0.75, True, 'classifier'), 'sel_50': (0.5, True, 'classifier'),
           'sel_50_global': (0.5, False, 'classifier'), 'sel_50_pixels': (0.5, True, 'pixels')}
RUNS = {'data_ref': (60000, '#333333'), 'sel_95': (57000, '#bcbd22'), 'sel_90': (54000, '#7f7f7f'), 'sel_75': (45000, '#1f77b4'), 'sel_50': (30000, '#ff7f0e'),
        'sel_50_global': (30000, '#d62728'), 'sel_50_pixels': (30000, '#9467bd'), 'noise_01': (60000, '#2ca02c'),
        'noise_03': (60000, '#17becf'), 'noise_01_const': (60000, '#8c564b'), 'sel_50_noise_01': (30000, '#e377c2')}


def real_subsets():
    """Ce que voit le GAN : chaque sous-ensemble comparé au MNIST test, comme une image générée."""
    x, y = mnist(train=True)
    with open(f'{OUT}/real_subsets.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'n', 'fid', 'precision', 'recall', 'classes_pct'])
        for name, (keep, per_class, space) in {'all': (1., True, 'classifier'), **SUBSETS}.items():
            idx = select_instances(y, embed(x, space, DEVICE), keep, per_class)
            rng = torch.Generator().manual_seed(0)  # même taille pour tous : les métriques en dépendent
            sub = idx[torch.randperm(len(idx), generator=rng)[:10000]]
            m = evaluate(x[sub], DEVICE)
            pct = 100 * torch.bincount(y[idx], minlength=10) / len(idx)
            row = [name, len(idx), f"{m['fid']:.1f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                   ' '.join(f'{p:.1f}' for p in pct)]
            w.writerow(row)
            print(row, flush=True)


def extremes():
    """Pour chaque chiffre : les 10 images les plus denses (gauche) et les 10 moins denses (droite)."""
    x, y = mnist(train=True)
    f, rows = embed(x, 'classifier', DEVICE), []
    for c in range(10):
        g = (y == c).nonzero().squeeze(1)
        order = g[gaussian_score(f[g]).argsort(descending=True)]
        rows.append(torch.cat([order[:10], order[-10:]]))
    grid = torchvision.utils.make_grid(x[torch.cat(rows)], nrow=20, padding=1, pad_value=0.5)
    torchvision.utils.save_image(grid, f'{OUT}/kept_vs_rejected.png')


def load_runs():
    """Courbes de chaque run depuis TensorBoard, regroupées par variante."""
    out = defaultdict(list)
    for name, (n_images, _) in RUNS.items():
        for run in sorted(glob.glob(f'runs/{name}_s[0-9]')):
            ea = EventAccumulator(run, size_guidance={'scalars': 0, 'histograms': 0})
            ea.Reload()
            curves = {k: {e.step: e.value for e in ea.Scalars(f'eval/{k}')} for k in ('fid', 'precision', 'recall')}
            steps = sorted(curves['fid'])
            at = min(steps, key=lambda s: abs(s * n_images // 100 - BUDGET))  # évaluation la plus proche du budget
            hist = {e.step: e.histogram_value for e in ea.Histograms('eval/classes')}[at]
            classes = np.zeros(10)
            for limit, count in zip(hist.bucket_limit, hist.bucket):
                classes[int(np.clip(np.floor(limit + 0.49999), 0, 9))] += count
            out[name].append(dict(curves=curves, steps=steps, at=at, classes=100 * classes / classes.sum(),
                                  collapsed=curves['recall'][steps[-1]] < 0.01))
    return out


def stat(rs, key, last=False):
    v = np.array([r['curves'][key][r['steps'][-1] if last else r['at']] for r in rs])
    return v.mean(), v.std()


def write_summary(runs):
    """Moyennes sur les runs non effondrés, au budget commun et à la fin de l'entraînement."""
    with open(f'{OUT}/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'seeds', 'collapsed', 'fid_60k', 'fid_60k_std', 'precision_60k', 'recall_60k',
                    'fid_final', 'precision_final', 'recall_final', 'classes_60k'])
        for name, rs in runs.items():
            ok = [r for r in rs if not r['collapsed']]
            row = [name, len(rs), len(rs) - len(ok), f"{stat(ok, 'fid')[0]:.1f}", f"{stat(ok, 'fid')[1]:.1f}",
                   f"{stat(ok, 'precision')[0]:.3f}", f"{stat(ok, 'recall')[0]:.3f}",
                   f"{stat(ok, 'fid', True)[0]:.1f}", f"{stat(ok, 'precision', True)[0]:.3f}",
                   f"{stat(ok, 'recall', True)[0]:.3f}", ' '.join(f'{c:.1f}' for c in np.mean([r['classes'] for r in ok], 0))]
            w.writerow(row)
            print(row)


def plot_curves(runs):
    """FID et recall en fonction du nombre de mises à jour, moyenne des runs non effondrés."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rs in runs.items():
        ok = [r for r in rs if not r['collapsed']]
        for ax, key in zip(axes, ('fid', 'recall')):
            xs = [s * RUNS[name][0] // 100 for s in ok[0]['steps']]
            ax.plot(xs, np.mean([[r['curves'][key][s] for s in r['steps']] for r in ok], 0),
                    color=RUNS[name][1], label=f'{name} (n={len(ok)})')
    for ax in axes:
        ax.axvline(BUDGET, color='#999999', lw=0.8, ls='--')
    axes[0].set(xlabel='mises à jour', ylabel='FID', yscale='log')
    axes[1].set(xlabel='mises à jour', ylabel='recall')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/curves.png', dpi=150)


def class_mix(runs):
    """Part de chaque chiffre dans les images générées au budget commun, moyenne des runs non effondrés."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(runs)
    m = np.array([np.mean([r['classes'] for r in runs[n] if not r['collapsed']], 0) for n in names])
    im = ax.imshow(m, cmap='RdBu_r', vmin=0, vmax=20, aspect='auto')
    ax.set(xticks=range(10), yticks=range(len(names)), yticklabels=names, xlabel='chiffre prédit')
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, label='%')
    fig.tight_layout()
    fig.savefig(f'{OUT}/classes.png', dpi=150)


def gaps():
    """Écart entre classes : distance d'une image à sa plus proche voisine d'un autre chiffre / du même chiffre."""
    x, y = mnist(train=True)
    f = embed(x, 'classifier', DEVICE)
    with open(f'{OUT}/class_gaps.csv', 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(['subset', 'pixels_other', 'pixels_same', 'pixels_ratio', 'features_other', 'features_same',
                    'features_ratio'])
        for name, (keep, per_class, space) in {'all': (1., True, 'classifier'), **SUBSETS}.items():
            idx = select_instances(y, embed(x, space, DEVICE), keep, per_class)
            sub = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:10000]]
            row = [name]
            for v in (x.flatten(1), f):
                d = torch.cdist(v[sub[:3000]], v[sub]).float()
                same = y[sub[:3000], None] == y[sub][None]
                d[d == 0] = float('inf')
                other = d.masked_fill(same, float('inf')).min(1).values.mean().item()
                near = d.masked_fill(~same, float('inf')).min(1).values.mean().item()
                row += [f'{other:.2f}', f'{near:.2f}', f'{other / near:.3f}']
            w.writerow(row)
            print(row, flush=True)


def per_class_real():
    """Même mesure sur les vraies images sélectionnées : ce que le GAN peut atteindre au mieux."""
    x, y = mnist(train=True)
    xt, yt = mnist(train=False)
    clf = load_classifier(DEVICE)
    f_real, f_train = predict(clf, xt, DEVICE)[0], predict(clf, x, DEVICE)[0]
    with open(f'{OUT}/per_class_real.csv', 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(['subset', 'mean_fid_per_digit'])
        for name, (keep, per_class_, space) in {'all': (1., True, 'classifier'), **SUBSETS}.items():
            if not per_class_:
                continue
            idx = select_instances(y, embed(x, space, DEVICE), keep, True)
            w.writerow([name, f'{np.mean([fid(f_real[yt == c], f_train[idx][y[idx] == c]) for c in range(10)]):.1f}'])


def per_class():
    """Qualité à chiffre fixé : FID de chaque chiffre généré contre le même chiffre réel, runs non effondrés."""
    clf = load_classifier(DEVICE)
    xt, yt = mnist(train=False)
    f_real = predict(clf, xt, DEVICE)[0]
    res = defaultdict(list)
    for name, rs in load_runs().items():
        for run, r in zip(sorted(glob.glob(f'runs/{name}_s[0-9]')), rs):
            if r['collapsed']:
                continue
            set_seed(0)
            G = load_model(Generator(g_output_dim=784), f'{run}/G.pth', DEVICE)
            f_fake, pred = predict(clf, generate(G, 50000, DEVICE), DEVICE)
            counts = torch.bincount(pred, minlength=10)
            scores = [fid(f_real[yt == c], f_fake[pred == c]) for c in range(10) if counts[c] >= 500]
            res[name].append((np.mean(scores), int((counts >= 500).sum())))
            print(run, f'{np.mean(scores):.1f}', (counts >= 500).sum().item(), flush=True)
    with open(f'{OUT}/per_class_fid.csv', 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(['variant', 'seeds', 'mean_fid_per_digit', 'std', 'digits_with_500_samples'])
        for name, v in res.items():
            a = np.array(v)
            w.writerow([name, len(v), f'{a[:, 0].mean():.1f}', f'{a[:, 0].std():.1f}', f'{a[:, 1].mean():.1f}'])


def by_budget(runs):
    """Par run : FID à 60 000 mises à jour, FID final, meilleur FID et son moment, effondrement."""
    with open(f'{OUT}/runs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'seed', 'fid_60k', 'fid_final', 'fid_best', 'updates_best', 'collapsed'])
        for name, rs in runs.items():
            for seed, r in enumerate(rs):
                fids = r['curves']['fid']
                best = min(fids, key=fids.get)
                w.writerow([name, seed, f"{fids[r['at']]:.1f}", f"{fids[r['steps'][-1]]:.1f}", f'{fids[best]:.1f}',
                            best * RUNS[name][0] // 100, int(r['collapsed'])])


def saturation(runs):
    """G final de chaque run : saturation de la tanh de sortie et gradient que G reçoit de D."""
    from model import Discriminator
    rows = []
    for name in RUNS:
        for run, r in zip(sorted(glob.glob(f'runs/{name}_s[0-9]')), runs[name]):
            G, D = Generator(784), Discriminator(784)
            G.load_state_dict(torch.load(f'{run}/G.pth', map_location='cpu'))
            D.load_state_dict(torch.load(f'{run}/D.pth', map_location='cpu'))
            pre = []
            G.fc4.register_forward_hook(lambda m, i, o: pre.append(o))
            x = G(torch.randn(500, 100, generator=torch.Generator().manual_seed(0)))
            d = D(x)
            torch.nn.functional.binary_cross_entropy(d, torch.ones_like(d)).backward()  # perte de G
            grad = torch.sqrt(sum(p.grad.norm() ** 2 for p in G.parameters())).item()
            rows.append([run[5:], int(r['collapsed']), f'{pre[0].abs().mean():.1f}', f'{(x.abs() > 0.99).float().mean():.2f}',
                         f'{x.std(0).mean():.3f}', f'{d.mean():.2e}', f'{grad:.2e}'])
    with open(f'{OUT}/saturation.csv', 'w', newline='') as f:
        csv.writer(f).writerows([['run', 'collapsed', 'preact_abs', 'saturated_pixels', 'std_over_z', 'D_G_z', 'grad_G']] + rows)
    for c in (0, 1):
        v = np.array([[float(x) for x in row[2:]] for row in rows if row[1] == c])
        print('raté' if c else 'sain', len(v), 'min', v.min(0).round(3), 'max', v.max(0).round(3))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if '--real' in sys.argv:
        real_subsets()
        extremes()
        gaps()
        per_class_real()
    elif '--per-class' in sys.argv:
        per_class()
    elif '--saturation' in sys.argv:
        saturation(load_runs())
    else:
        runs = load_runs()
        write_summary(runs)
        by_budget(runs)
        plot_curves(runs)
        class_mix(runs)
