#!/bin/bash
# Lance toutes les expériences du volet Données (configs/data/) : 6 runs en parallèle, seeds 0 à 2.
# Peut être relancé : les runs terminés sont sautés, les runs interrompus reprennent.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=${PYTHON:-python}
one() {
    name=$(basename "$1" .yaml); run=runs/${name}_s$2
    if grep -q ",${name}_s$2," results.csv 2>/dev/null; then return; fi
    mkdir -p logs
    if [ -f "$run/ckpt/last.pth" ]; then resume=--resume; else rm -rf "$run"; resume=; fi
    $PYTHON train.py --config "$1" --seed "$2" $resume >> "logs/${name}_s$2.log" 2>&1
}
export -f one; export PYTHON
for seed in 0 1 2; do for c in configs/data/*.yaml; do echo "$c $seed"; done; done |
    xargs -P 6 -L 1 bash -c 'one "$0" "$1"'
