#!/bin/bash
# Lance les expériences du volet Données (configs/data/, seeds 0 à 2) sur Juliet, 4 jobs à la fois.
# Peut être relancé : un run terminé est sauté, un run coupé reprend.
mkdir -p outputs
n=$(ls configs/data/*.yaml | wc -l)
sbatch --array=0-$((3 * n - 1))%4 <<'EOT'
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=gan_data
#SBATCH --output=outputs/%x_%A_%a.out
#SBATCH --error=outputs/%x_%A_%a.err
export DATA=/projects/m25146/data/
source venv/bin/activate
configs=(configs/data/*.yaml)
config=${configs[SLURM_ARRAY_TASK_ID % ${#configs[@]}]}
seed=$((SLURM_ARRAY_TASK_ID / ${#configs[@]}))
run=$(basename "$config" .yaml)_s$seed
grep -q ",$run," results.csv 2>/dev/null && exit 0
[ -f "runs/$run/ckpt/last.pth" ] && resume=--resume || rm -rf "runs/$run"
python train.py --config "$config" --seed "$seed" $resume
EOT
