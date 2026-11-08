# Lancés sur Juliet le 8/11

À copier dans un clone de `noe/training` : `juliet2.sh` et `drs_recalib.py` dans `scripts/`, les YAML dans `configs/juliet2/`.

- `bash scripts/juliet2.sh` : baseline + EMA (3 seeds) et baseline + bruit 0,3 + EMA (5 seeds), puis mesure → `results/juliet2_test.csv`.
- `scripts/run.sh "PYTHONPATH=. python scripts/drs_recalib.py results/drs_recalib_test.csv" 03:00:00` : DRS avec D recalibré, en Inception.
- `scripts/run.sh "PYTHONPATH=. python scripts/kid_perclass.py outputs/kid_manifest.csv results/kid_perclass_test.csv" 03:00:00` : KID Inception et FID par chiffre (Inception et CNN), modèles du protocole commun.
- `scripts/run.sh "PYTHONPATH=. python scripts/sensitivity_10k.py results/sensitivity_10k.csv" 03:00:00` : tests de sensibilité à 10 000 images (FID et KID Inception, FID CNN).
- Mesure des runs Données de Nicodème : `git archive origin/nico/preprocessing runs | tar -x -C nico_runs`, un manifeste `nom,nico_runs/runs/<run>/G.pth`, puis `scripts/measure_runs.py` → `results/nico_data_test.csv`.
