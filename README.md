# DataLabAssignement2

## Using GPUs with MesoNet

### 1. Access MesoNet
To use the provided GPUs, you must first:
- Create a MesoNet account and request access to the project.
- Once accepted, add your SSH key to MesoNet for secure access. Follow the instructions [here](https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion).

### 2. Set Up Your Environment
On Juliet (MesoNet's cluster), you need to:

1. Create a virtual environment for Python:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training or Generation
- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- If you don't use Juliet cluster, please provide a data_path argument for the training scripts. Otherwise, everything should be automated.

### 4. Hardware Compatibility
The code is compatible with:
- **NVIDIA GPUs (CUDA)**
- **Apple Silicon (MPS, for M1/M2/M3/M4 chips)**
- **CPU-only mode** (not recommended for performance)

We recommend using CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon).

### 5. Slurm Resources
- If you need help with Slurm (MesoNet's job scheduler), there will be a Slurm lecture on 29/10/2025 or refer to the official documentation: [Slurm Documentation](https://slurm.schedmd.com/documentation.html).

### 6. Example Commands
To submit a job to MesoNet:
```bash
# Make the script executable
chmod +x scripts/train.sh

# Submit the job to Slurm
scripts/train.sh
```
## Workflow du groupe

Choix et justifications détaillés : [architecture-choix.md](architecture-choix.md).

| Étape | Commande |
|---|---|
| Entraîner | `python train.py --config configs/baseline.yaml` (sorties dans `runs/<name>/`, `--resume` pour reprendre) |
| Suivre | `tensorboard --logdir runs` |
| Évaluer | `python evaluate.py --checkpoint runs/<name>/G.pth` ou `--samples samples` (FID, precision, recall → `results.csv`) |
| Tuner | `python tune.py --config configs/baseline.yaml --n-trials 20 --epochs 20` (Optuna, `runs/optuna.db`) |
| Rendre | `cp runs/<name>/G.pth checkpoints/G.pth` puis `python generate.py` |

Sur Juliet : `scripts/train.sh [config]`, `scripts/tune.sh [config] [n_trials]`, `scripts/generate.sh` (`DATA` y est déjà configuré). Hors Juliet, MNIST est téléchargé dans `data/`. Les hyperparamètres se règlent dans le YAML (`configs/`), pas dans les scripts.

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples, from `checkpoints/G.pth` (or `--checkpoint <path>`). Outputs are rescaled to [0, 1]. You can launch this script on juliet with `scripts/generate.sh`

Example:
  > python3 generate.py --batch_size 64

or 

```bash
# Make the script executable
chmod +x scripts/generate.sh

# Submit the job to Slurm
scripts/generate.sh
```

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.

