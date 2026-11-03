"""Campagne du volet Entraînement, lancée depuis la racine du repo :

    nohup caffeinate -i python3 scripts/sweep_training.py > logs/sweep_training.log 2>&1 &

Phase 1 : 3 seeds x {baseline, RpGAN + R1, R3GAN}, 100 époques, EMA 0.999.
Phase 2 : sur la meilleure perte de la phase 1 (FID moyen sur les seeds), balayage de gamma (R1/R2) in {0.1, 10}.
Chaque run écrit runs/<name>/, logs/<name>.log et une ligne dans results.csv ; un run déjà présent est ignoré.
"""
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

SEEDS = (0, 1, 2)
GAMMAS = (0.1, 10.0)
N_PARALLEL = 6
COMMON = {'train': {'epochs': 100, 'batch_size': 100, 'num_workers': 2}, 'ema': {'decay': 0.999},
          'output': {'save_every': 50}}  # checkpoints espacés : chaque ckpt pèse ~40 Mo
LOSSES = {
    'baseline': {'name': 'bce'},
    'rpgan_r1': {'name': 'rpgan', 'r1_gamma': 1.0, 'r2_gamma': 0.0},
    'r3gan': {'name': 'rpgan', 'r1_gamma': 1.0, 'r2_gamma': 1.0},
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def make_config(name, loss, seed):
    with open('configs/baseline.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg.update(name=name, seed=seed, loss=loss)
    cfg['train'].update(COMMON['train'])
    cfg['ema'] = dict(COMMON['ema'])
    cfg['output'].update(COMMON['output'])
    path = f'configs/sweep/{name}.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run(name, loss, seed):
    if os.path.exists(f'runs/{name}/G.pth') and os.path.exists(f'runs/{name}/grid.png'):
        log(f'{name} : déjà fait, ignoré')
        return
    path = make_config(name, loss, seed)
    # run interrompu (dossier + last.pth présents) : reprise, log complété au lieu d'être écrasé
    resume = os.path.exists(f'runs/{name}/ckpt/last.pth')
    log(f'{name} : {"reprise" if resume else "démarrage"}')
    with open(f'logs/{name}.log', 'a' if resume else 'w') as out:
        cmd = [sys.executable, 'train.py', '--config', path] + (['--resume'] if resume else [])
        code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
    log(f'{name} : terminé (code {code})')


def run_all(jobs):
    with ThreadPoolExecutor(N_PARALLEL) as pool:
        list(pool.map(lambda j: run(*j), jobs))


def final_fid(name):
    rows = [r for r in csv.DictReader(open('results.csv')) if r['run'] == name]
    return float(rows[-1]['fid']) if rows else float('inf')


if __name__ == '__main__':
    os.makedirs('configs/sweep', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    log('Phase 1 : 3 seeds x {baseline, rpgan_r1, r3gan}, 100 époques, EMA')
    run_all([(f'ema100_{k}_s{s}', loss, s) for k, loss in LOSSES.items() for s in SEEDS])

    means = {k: sum(final_fid(f'ema100_{k}_s{s}') for s in SEEDS) / len(SEEDS) for k in ('rpgan_r1', 'r3gan')}
    best = min(means, key=means.get)
    log(f'FID moyens : {means} -> balayage de gamma sur {best}')

    log(f'Phase 2 : gamma in {GAMMAS} sur {best}, 3 seeds')
    jobs = []
    for g in GAMMAS:
        loss = dict(LOSSES[best], r1_gamma=g, r2_gamma=g if LOSSES[best]['r2_gamma'] else 0.0)
        jobs += [(f'ema100_{best}_g{g:g}_s{s}', loss, s) for s in SEEDS]
    run_all(jobs)
    log('Campagne terminée')
