"""Recherche d'hyperparamètres avec Optuna : minimise le FID, precision / recall gardés pour l'analyse.

Chaque essai est un run normal (runs/<study>/trial_XXX/, visible dans TensorBoard).
L'étude est stockée dans runs/optuna.db : relancer la même commande la reprend.
"""
import argparse
import copy
import os
import optuna

from train import train
from utils import load_config


def objective(trial, base, study, epochs):
    cfg = copy.deepcopy(base)
    cfg['name'] = os.path.join(study, f'trial_{trial.number:03d}')
    cfg['train']['epochs'] = epochs
    cfg['eval']['every'] = max(1, epochs // 4)  # 4 rapports intermédiaires pour le pruner
    # Espace de recherche
    cfg['optimizer']['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    cfg['optimizer']['betas'][0] = trial.suggest_categorical('beta1', [0.0, 0.5, 0.9])
    cfg['train']['batch_size'] = trial.suggest_categorical('batch_size', [64, 100, 128])

    m = train(cfg, trial=trial)
    trial.set_user_attr('precision', m['precision'])
    trial.set_user_attr('recall', m['recall'])
    return m['fid']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search with Optuna.')
    parser.add_argument('--config', default='configs/baseline.yaml', help='Config de base.')
    parser.add_argument('--study', default='tune_baseline', help='Nom de l\'étude (dossier runs/<study>/).')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20, help='Budget d\'époques identique pour chaque essai.')
    args = parser.parse_args()

    base = load_config(args.config)
    os.makedirs(base['output']['dir'], exist_ok=True)
    study = optuna.create_study(
        study_name=args.study, direction='minimize', load_if_exists=True,
        storage=f"sqlite:///{os.path.join(base['output']['dir'], 'optuna.db')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    # Graine décalée par le nombre d'essais déjà faits : reproductible sans reproposer les mêmes valeurs à la reprise
    study.sampler = optuna.samplers.TPESampler(seed=base['seed'] + len(study.trials))
    study.optimize(lambda t: objective(t, base, args.study, args.epochs), n_trials=args.n_trials)

    for t in study.trials:
        fid = f'{t.value:.3f}' if t.value is not None else '-'
        print(f'#{t.number:03d} {t.state.name:8s} FID={fid} {t.user_attrs} {t.params}')
    best = study.best_trial
    print(f'Meilleur essai #{best.number} : FID={best.value:.3f} '
          f"P={best.user_attrs['precision']:.3f} R={best.user_attrs['recall']:.3f} | {best.params}")
