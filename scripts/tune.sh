#!/bin/bash
# Usage : scripts/tune.sh [configs/xxx.yaml] [n_trials]   (relancer reprend l'étude dans runs/optuna.db)
config=${1:-configs/baseline.yaml}
n_trials=${2:-20}
outdir="outputs"
export DATA="/projects/m25146/data/"
mkdir -p ${outdir}

    echo "Launching Optuna search with ${config} (${n_trials} trials)"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_tune      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source venv/bin/activate
python tune.py --config ${config} --n-trials ${n_trials}
EOT
