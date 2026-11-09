#!/bin/bash
# Étude d'inférence complète sur Juliet : plain, OBRS(K), DRS(K), truncation(psi), 3 seeds chacun.
# Les lignes s'ajoutent à results/results.csv.
# Usage : scripts/sweep.sh [run_dir] [weights] [reference]
#         scripts/sweep.sh runs/baseline_200ep_prof best val      # choisir K / psi
#         scripts/sweep.sh runs/baseline_200ep_prof best test     # chiffres finaux du rapport
run=${1:-checkpoints}
weights=${2:-last}
reference=${3:-val}
outdir="outputs"
export DATA="/projects/m25146/data/"
mkdir -p ${outdir}

    echo "Launching inference sweep on ${run} (${weights}, reference ${reference})"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146        # Your project account
#SBATCH --job-name=gan_sweep      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
set -euo pipefail
source venv/bin/activate
common="--run ${run} --weights ${weights} --reference ${reference} --seeds 0 1 2"
python evaluate.py \${common} --sampling configs/sampling/plain.yaml
python sweep.py \${common} --sampling configs/sampling/obrs.yaml --param K --values 1.5 2 3 5 10 20 50
python sweep.py \${common} --sampling configs/sampling/drs.yaml --param K --values 1.5 2 3 5 10 20 50
python sweep.py \${common} --sampling configs/sampling/truncation.yaml --param psi --values 0.5 1 1.5 2 2.5 3
EOT
