#!/bin/bash
# Usage : scripts/train.sh [configs/xxx.yaml]
config=${1:-configs/baseline.yaml}
outdir="outputs"
n_gpu=1
export DATA="/projects/m25146/data/"
mkdir -p ${outdir}

    echo "Launching training with ${config}"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_train      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source venv/bin/activate
# Ajouter --resume pour reprendre un run coupé par la limite de temps
python train.py --config ${config}
EOT
