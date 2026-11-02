"""Fait varier un paramètre d'une méthode d'échantillonnage et évalue chaque valeur sur plusieurs seeds.

Chaque (valeur, seed) ajoute une ligne à results/results.csv ; plot_results.py en tire les figures.
La référence est 'val' par défaut : c'est sur elle que l'on choisit K ou psi.

Utilisation :
    python sweep.py --run checkpoints --sampling configs/sampling/obrs.yaml --param K --values 1 2 5 10 20 --seeds 0 1 2
    python sweep.py --run checkpoints --sampling configs/sampling/drs.yaml  --param K --values 2 5 10 20
    python sweep.py --run checkpoints --sampling configs/sampling/truncation.yaml --param psi --values 0.5 1 1.5 2 3
"""
import argparse

import yaml

from evaluate import EVALUATORS, WEIGHT_FILES, append_result, evaluate_sampling, format_summary, load_run, summarize
from utils import get_device, load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep one parameter of a sampling method.')
    parser.add_argument('--run', required=True, help='Dossier contenant G.pth / D.pth (runs/<name> ou checkpoints).')
    parser.add_argument('--weights', choices=sorted(WEIGHT_FILES), default='last', help='Jeu de poids G et D du run.')
    parser.add_argument('--sampling', required=True, help='Config de base de la méthode (YAML).')
    parser.add_argument('--param', required=True, help='Nom du paramètre à faire varier (ex. K, psi).')
    parser.add_argument('--values', nargs='+', required=True, help='Valeurs du paramètre.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seeds de l\'échantillonnage.')
    parser.add_argument('--reference', choices=['val', 'test'], default='val', help='Split réel de référence.')
    parser.add_argument('--metric', choices=sorted(EVALUATORS), default='cnn', help='cnn : petit CNN ; prof : Inception et VGG.')
    args = parser.parse_args()
    device = get_device()

    G, D = load_run(args.run, args.weights, device)
    for value in map(yaml.safe_load, args.values):          # '10' -> 10, '0.5' -> 0.5
        cfg = {**load_config(args.sampling), args.param: value}
        rows = []
        for seed in args.seeds:
            row, _ = evaluate_sampling(G, D, cfg, seed, args.reference, device, args.run, args.weights, args.metric)
            append_result(row, args.metric)
            rows.append(row)
        for key, agg in summarize(rows).items():             # une seule clé : cette valeur, toutes seeds
            print(format_summary(key, agg), flush=True)
