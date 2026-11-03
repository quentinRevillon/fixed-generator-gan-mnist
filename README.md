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

## Volet Entraînement : perte R3GAN

**Méthode :** on remplace la perte GAN d'origine par celle de **R3GAN**, une perte relativiste (RpGAN) avec les pénalités de gradient R1 et R2, tirée de Huang et al., *The GAN is dead; long live the GAN! A Modern GAN Baseline*, NeurIPS 2024 ([arXiv:2501.05441](https://arxiv.org/abs/2501.05441)). **Le générateur et `model.py` ne sont pas modifiés :** seule la façon de l'entraîner change.

![Exemples : vraies images, baseline, RpGAN + R1 avec le même bruit z](figures/examples_best.png)

**Résultats finaux** (évaluation du groupe ; 100 époques, batch 100, EMA 0.999, **3 seeds** par configuration) :

| Configuration | FID ↓ | Precision ↑ | Recall ↑ | Seeds stables |
|---|---|---|---|---|
| Baseline (perte BCE d'origine) | 58,7 ± 4,8 | **0,80** | 0,79 | 2 / 3 |
| **RpGAN + R1, γ = 1** | **14,0 ± 0,9** | 0,72 | 0,80 | **3 / 3** |
| R3GAN (RpGAN + R1 + R2), γ = 1 | 14,4 ± 0,9 | 0,70 | 0,80 | 3 / 3 |
| RpGAN + R1, γ = 10 | 17,7 ± 0,9 | 0,67 | 0,79 | 3 / 3 |
| RpGAN + R1, γ = 0,1 | diverge | — | — | 0 / 3 |

- **Le FID est divisé par 4** et l'entraînement devient fiable : aucune seed effondrée, contre 1 sur 3 pour la baseline.
- **Le recall est le même partout.** La baseline couvre tous les chiffres, mais dans de mauvaises proportions (4 % de « 2 », 16 % de « 1 »). La perte relativiste corrige ces proportions (8 à 12 % par chiffre), ce que le FID mesure et que la precision et le recall par plus proches voisins ne voient pas.
- **Le prix à payer :** un peu de precision (0,80 → 0,72), que le rejet du volet Inférence peut récupérer.
- **R2 n'apporte rien de mesurable**, γ = 1 est le bon réglage, et sans R1 la perte relativiste diverge.

Le meilleur générateur est `runs/ema100_rpgan_r1_s1/G.pth` (FID 13,1). L'explication complète (papier, formules, première série à 50 époques, courbes, répartition des chiffres, limites) est dans **[training-r3gan.md](training-r3gan.md)**.

**Utilisation :** deux sections facultatives du YAML. Sans elles, l'entraînement est exactement la baseline.

```yaml
loss:
  name: rpgan      # 'bce' = baseline (défaut)
  r1_gamma: 1.0    # pénalité R1 (gradient de D sur les vraies images)
  r2_gamma: 0.0    # pénalité R2 (gradient de D sur les fausses images)
ema:
  decay: 0.999     # moyenne mobile des poids de G : c'est elle qui est sauvegardée dans G.pth
```

```bash
python train.py --config configs/sweep/ema100_rpgan_r1_s1.yaml       # un run
python scripts/sweep_training.py                                       # toute la campagne (3 seeds + balayage de γ)
```

| Fichiers ajoutés | Contenu |
|---|---|
| `configs/train_*.yaml`, `configs/baseline_rerun.yaml` | première série (50 époques, seed 0) |
| `configs/sweep/` | campagne finale : une config par run (100 époques, EMA, seeds 0-2, γ) |
| `runs/<config>/G.pth`, `runs/<config>/grid.png` | générateur entraîné et grille de 100 images de chaque run |
| `logs/<config>.log` | logs d'entraînement (pertes, FID / P / R toutes les 10 époques) |
| `results.csv` | une ligne par évaluation |
| `figures/` | figures de [training-r3gan.md](training-r3gan.md) et scripts `make_*_figures.py` pour les régénérer |

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

