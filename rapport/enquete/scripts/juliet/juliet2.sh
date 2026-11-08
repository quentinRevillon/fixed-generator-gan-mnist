#!/bin/bash
# Juliet, 8/11 : cases manquantes du protocole commun (EMA sur la baseline, bruit 0.3 sur la baseline), puis mesure.
# Usage, depuis la racine du repo : bash scripts/juliet2.sh
mkdir -p outputs
runs=(baseline_ema:0 baseline_ema:1 baseline_ema:2 baseline_noise_03_ema:0 baseline_noise_03_ema:1 baseline_noise_03_ema:2 baseline_noise_03_ema:3 baseline_noise_03_ema:4)
train_job=$(sbatch --parsable --array=0-$((${#runs[@]} - 1))%4 <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=gan_j2
#SBATCH --output=outputs/%x_%A_%a.out
#SBATCH --error=outputs/%x_%A_%a.err
export DATA=/projects/m25146/data/
source venv/bin/activate
runs=(${runs[*]})
item=\${runs[SLURM_ARRAY_TASK_ID]}
python train.py --config configs/juliet2/\${item%%:*}.yaml --seed \${item##*:}
EOT
)
manifest=outputs/j2_manifest.csv
: > $manifest
for item in "${runs[@]}"; do
  r=runs/j2_${item%%:*}_s${item##*:}
  echo "${item%%:*} G_EMA,$r/G.pth" >> $manifest
  echo "${item%%:*} G brut,$r/G_raw.pth" >> $manifest
  echo "${item%%:*} G_EMA + DRS,$r/G.pth,$r/D.pth" >> $manifest
  echo "${item%%:*} G brut + DRS,$r/G_raw.pth,$r/D.pth" >> $manifest
done
for s in 0 1 2; do
  for c in rpgan_r1_ema rpgan_r1_ema_noise_03; do
    r=runs/cmploc_${c}_s$s
    echo "cmploc_$c G brut,$r/G_raw.pth" >> $manifest
    echo "cmploc_$c G brut + DRS,$r/G_raw.pth,$r/D.pth" >> $manifest
  done
done
sbatch --dependency=afterany:${train_job} <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=gan_j2_measure
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
export DATA=/projects/m25146/data/
source venv/bin/activate
PYTHONPATH=. python scripts/measure_runs.py $manifest results/juliet2_test.csv
EOT
echo "train job: $train_job"
