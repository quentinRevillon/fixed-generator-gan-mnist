#!/bin/bash
# Lance une commande quelconque sur un GPU de Juliet, à lancer depuis la racine du repo.
# Usage : scripts/run.sh "<commande>" [limite de temps]
#         scripts/run.sh "python evaluate.py --run checkpoints --metric prof --reference test" 00:20:00
command=$1
limit=${2:-00:20:00}
outdir="outputs"
export DATA="/projects/m25146/data/"
mkdir -p ${outdir}

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=${limit}
#SBATCH --mem=32G
#SBATCH --account=m25146        # Your project account
#SBATCH --job-name=gan_run      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
set -euo pipefail
source venv/bin/activate
${command}
EOT
