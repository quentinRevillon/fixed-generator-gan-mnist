# Lancés sur Juliet le 8/11

À copier dans un clone de `noe/training` : `juliet2.sh` et `drs_recalib.py` dans `scripts/`, les YAML dans `configs/juliet2/`.

- `bash scripts/juliet2.sh` : baseline + EMA (3 seeds) et baseline + bruit 0,3 + EMA (5 seeds), puis mesure → `results/juliet2_test.csv`.
- `scripts/run.sh "PYTHONPATH=. python scripts/drs_recalib.py results/drs_recalib_test.csv" 03:00:00` : DRS avec D recalibré, en Inception.
