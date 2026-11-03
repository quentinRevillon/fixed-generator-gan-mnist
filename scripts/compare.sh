#!/bin/bash
# Compare les méthodes à la baseline commune sur Juliet : même split, même nombre de mises à jour, seeds 0 à 2.
# Peut être relancé : un run terminé est sauté, un run coupé reprend.
# Usage, depuis la racine du repo : bash scripts/compare.sh
mkdir -p outputs
n=$(ls configs/compare/*.yaml | wc -l)
train_job=$(sbatch --parsable --array=0-$((3 * n - 1))%4 <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=gan_cmp
#SBATCH --output=outputs/%x_%A_%a.out
#SBATCH --error=outputs/%x_%A_%a.err
export DATA=/projects/m25146/data/
source venv/bin/activate
configs=(configs/compare/*.yaml)
config=${configs[SLURM_ARRAY_TASK_ID % ${#configs[@]}]}
seed=$((SLURM_ARRAY_TASK_ID / ${#configs[@]}))
run=cmp_$(basename "$config" .yaml)_s$seed
[ -f "runs/$run/grid.png" ] && exit 0
[ -f "runs/$run/ckpt/last.pth" ] && resume=--resume || rm -rf "runs/$run"
python train.py --config "$config" --seed "$seed" $resume
EOT
)

# Une fois tous les entraînements finis : même mesure pour tous, contre MNIST test, écrite dans results/comparison_test.csv.
sbatch --dependency=afterany:${train_job} <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=gan_cmp_measure
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
export DATA=/projects/m25146/data/
source venv/bin/activate
manifest=outputs/comparison_manifest.csv
: > "$manifest"
for s in 0 1 2; do
  echo "Baseline commune,runs/cmp_baseline_s$s/G_best.pth" >> "$manifest"
  echo "DRS,runs/cmp_baseline_s$s/G_best.pth,runs/cmp_baseline_s$s/D_best.pth" >> "$manifest"
  echo "Bruit,runs/cmp_noise_01_s$s/G_best.pth" >> "$manifest"
  echo "Selection,runs/cmp_sel_50_s$s/G_best.pth" >> "$manifest"
  echo "RpGAN + R1,runs/cmp_rpgan_r1_s$s/G_best.pth" >> "$manifest"
  echo "R3GAN,runs/cmp_r3gan_s$s/G_best.pth" >> "$manifest"
  echo "R3GAN + DRS,runs/cmp_r3gan_s$s/G_best.pth,runs/cmp_r3gan_s$s/D_best.pth" >> "$manifest"
done
PYTHONPATH=. python scripts/measure_runs.py "$manifest" results/comparison_test.csv
EOT
