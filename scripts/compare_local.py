"""Pipeline combiné des trois volets, sur une machine locale (Mac MPS ou un GPU), depuis la racine du repo :

    nohup caffeinate -i python3 scripts/compare_local.py > logs/compare_local.log 2>&1 &

1. Entraîne CONFIGS x SEEDS (split commun, 200 époques sur train = même nombre de mises à jour que la baseline
   commune). Pendant l'entraînement, seul le petit CNN est mesuré, sur val : Inception et VGG prennent 10 à 20 min
   par mesure sur un Mac. Un run terminé (grid.png présent) est sauté, un run coupé reprend avec --resume : on peut
   relancer le script sans risque.
2. Mesure les poids de la dernière époque (G.pth, le G moyenné si EMA, et D.pth) contre MNIST test, sans tri et avec
   le DRS au percentile 80, avec scripts/measure_runs.py : FID Inception, précision et rappel VGG, et les mêmes
   mesures du petit CNN. On ne choisit pas d'époque : la baseline commune est meilleure à l'époque 200 selon
   Inception (docs/baseline.md), et le petit CNN choisirait une autre époque. Sortie : results/comparison_local.csv.

C'est l'équivalent local de scripts/compare.sh (Slurm, Juliet), réduit aux configurations du pipeline combiné.
Sur Juliet, utiliser configs/compare/ (mêmes réglages, Inception et VGG mesurés pendant l'entraînement).
"""
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CONFIGS = {   # nom affiché -> config ; le DRS est mesuré sur chacune
    'Baseline commune': 'configs/compare_local/baseline.yaml',
    'RpGAN + R1 + EMA': 'configs/compare_local/rpgan_r1_ema.yaml',
    'RpGAN + R1 + EMA + bruit 0,3': 'configs/compare_local/rpgan_r1_ema_noise_03.yaml',
}
SEEDS = (0, 1, 2)
N_PARALLEL = int(os.getenv('N_PARALLEL', 3))
OUT = 'results/comparison_local.csv'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run_name(config, seed):
    """Nom du run tel que train.py --seed le construit : <name du YAML>_s<seed>."""
    with open(config) as f:
        name = next(line.split(':', 1)[1].strip() for line in f if line.startswith('name:'))
    return f'{name}_s{seed}'


def train(job):
    config, seed = job
    run = run_name(config, seed)
    if os.path.exists(f'runs/{run}/grid.png'):
        log(f'{run} : déjà terminé')
        return run, 0
    cmd = [sys.executable, 'train.py', '--config', config, '--seed', str(seed)]
    if os.path.exists(f'runs/{run}/ckpt/last.pth'):
        cmd.append('--resume')
    log(f'{run} : démarrage {"(reprise)" if "--resume" in cmd else ""}')
    with open(f'logs/{run}.log', 'a') as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f'{run} : terminé (code {code})')
    return run, code


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    jobs = [(config, seed) for seed in SEEDS for config in CONFIGS.values()]
    log(f'Entraînement : {len(jobs)} runs, {N_PARALLEL} en parallèle')
    with ThreadPoolExecutor(N_PARALLEL) as pool:
        codes = dict(pool.map(train, jobs))
    failed = [run for run, code in codes.items() if code != 0]
    if failed:
        log(f'Runs en échec : {failed}. Relancer le script pour les reprendre ; la mesure est faite sur les autres.')

    manifest = 'outputs/comparison_local_manifest.csv'
    os.makedirs('outputs', exist_ok=True)
    with open(manifest, 'w', newline='') as f:
        rows = csv.writer(f, lineterminator='\n')   # les noms peuvent contenir des virgules (« bruit 0,3 »)
        for seed in SEEDS:
            for label, config in CONFIGS.items():
                run = run_name(config, seed)
                if codes.get(run) != 0:
                    continue
                rows.writerow([label, f'runs/{run}/G.pth'])
                rows.writerow([f'{label} + DRS', f'runs/{run}/G.pth', f'runs/{run}/D.pth'])
    log(f'Mesure contre MNIST test -> {OUT}')
    code = subprocess.run([sys.executable, 'scripts/measure_runs.py', manifest, OUT],
                          env={**os.environ, 'PYTHONPATH': '.'}).returncode
    log(f'Campagne terminée (mesure : code {code})')
