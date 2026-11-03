# A2 : GAN MNIST à générateur fixe (groupe 3)

On entraîne le générateur imposé (`model.py`, architecture **non modifiable**) sur MNIST, puis on étudie trois leviers (slide 76) :
- **Données** (Nicodème) ;
- **Entraînement** (Noé) ;
- **Inférence** (Quentin).

On mesure leur effet sur la qualité et la diversité : FID, précision, rappel et inspection visuelle.

## Où trouver quoi

| Je cherche | Document | Branche |
|---|---|---|
| La baseline : réglages, résultats, vérifications | `baseline.md` | `quentin/baseline` |
| Le volet Inférence : DRS, OBRS, troncature | `inference-drs.md` | `quentin/inference` |
| Le volet Données | `rapport-donnée.md` | `nico/preprocessing` |
| Le volet Entraînement | `training-r3gan.md` | `noe/training` |
| Les choix d'organisation et leurs sources | [architecture-choix.md](architecture-choix.md) | toutes |
| Les slides | `slides/presentation.tex` | `quentin/inference` |

## Les branches

| Branche | Contenu | Part de |
|---|---|---|
| `main` | la version commune du groupe | |
| `quentin/baseline` | split, reproductibilité, mesure du classement, baseline de référence | `main` |
| `quentin/inference` | la baseline, plus les méthodes d'inférence et leurs résultats | `quentin/baseline` |
| `nico/preprocessing` | volet Données | `main` |
| `noe/training` | volet Entraînement | `main` |

## Démarrage rapide

```bash
python -m venv venv && source venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q                          # quelques secondes, sans GPU
python generate.py --batch_size 256                # 10 000 images dans samples/ (interface imposée)
python evaluate.py --samples samples               # les évaluer comme le correcteur
```

Hors de Juliet, MNIST est téléchargé dans `data/`. Sur Juliet, `$DATA=/projects/m25146/data/` est défini par les scripts Slurm.

## Les commandes

| Étape | Commande | Sorties |
|---|---|---|
| Entraîner | `python train.py --config configs/baseline.yaml` (`--resume` pour reprendre) | `runs/<name>/` |
| Suivre | `tensorboard --logdir runs` | courbes, grilles |
| Évaluer un run | `python evaluate.py --run runs/<name> --weights best --seeds 0 1 2 --metric prof` | lignes dans `results/results_prof.csv` (`--metric cnn` : `results/results.csv`) |
| Faire varier un paramètre (`quentin/inference`) | `python sweep.py --run runs/<name> --weights best --sampling configs/sampling/drs_azadi.yaml --param percentile --values 50 80 95 99 --metric prof` | idem |
| Tracer les figures | `python figures/make_baseline_figures.py --weights <dossier>` ; `python figures/make_inference_figures.py` | `figures/baseline/`, `figures/inference/` |
| Tuner | `python tune.py --config configs/baseline.yaml --n-trials 20 --epochs 20` | `runs/optuna.db` |
| Rendre | copier `G_best.pth` / `D_best.pth` dans `checkpoints/G.pth` / `D.pth`, écrire la méthode dans `checkpoints/sampling.yaml` | `checkpoints/` |

**Sur Juliet** (Slurm, compte `m25146`) :
- `bash scripts/run.sh "<commande>" [limite]` lance n'importe quelle commande sur un GPU, par exemple `bash scripts/run.sh "python train.py --config configs/baseline_200ep_prof.yaml" 01:30:00` ;
- `scripts/train.sh [config]` (limite de 20 minutes) ;
- `scripts/tune.sh [config] [n_trials]` ;
- `scripts/sweep.sh [run] [weights] [reference]` lance l'étude d'inférence complète (`quentin/inference`) ;
- `scripts/generate.sh`.

Créer le venv `venv/` à la racine avant de lancer ces scripts.

## Les données : un split commun figé

| Split | Taille | Sert à |
|---|---|---|
| `train` | 50 000 (MNIST train) | entraîner G et D |
| `val` | 10 000 (MNIST train, permutation seed 0) | choisir : époque, hyperparamètres, K, ψ |
| `test` | 10 000 (MNIST test) | chiffres finaux du rapport uniquement |

Défini dans `data.py`. C'est un contrat entre les trois volets : on ne le change pas sans prévenir le groupe.

## Organisation du code

```
model.py              G et D imposés (non modifiés)
utils.py, train.py, evaluate.py, generate.py, tune.py   les fichiers de Nicodème (structure à plat)
data.py               split train / val / test
sampling.py           comment tirer des images : plain sur quentin/baseline ; truncation, DRS, OBRS en plus sur quentin/inference
keep_training.py, sweep.py   étape KeepTraining du DRS, variation d'un paramètre (quentin/inference)
configs/              une config YAML par expérience (entraînement) ou par méthode (configs/sampling/)
checkpoints/          ce que rend generate.py : G.pth, D.pth, sampling.yaml, mnist_classifier.pth
runs/<name>/          un run (config, metrics.jsonl, poids utiles : versionnés)
results/              un tableau par mesure : results.csv (petit CNN), results_prof.csv (Inception et VGG)
figures/              les figures par volet, et les scripts make_*_figures.py qui les tracent
slides/               la présentation LaTeX et ses figures (quentin/inference)
tests/                tests unitaires : python -m pytest tests -q
```

Les runs de la baseline :

| Dossier | Contenu |
|---|---|
| `runs/baseline_200ep_prof/` | **La baseline de référence** : 200 époques, mesures toutes les 2 époques, poids de l'époque 200 |
| `runs/baseline_200ep/` | Premier passage, mêmes poids, mesures toutes les 10 époques. Conservé car le DRS sur l'époque 120 y renvoie |
| `runs/baseline_v1/` | Premier essai à 50 époques (split 50 000 / 10 000) |
| `runs/baseline/` | Baseline de Nicodème à 50 époques, sur 60 000 images |

## Reproductibilité

- **Même config + même machine ⇒ mêmes poids au bit près**, même après `--resume`, et quelle que soit la fréquence d'évaluation. Vérifié sur 200 époques entre deux nœuds A100 de Juliet.
- `generate.py` et toutes les méthodes d'échantillonnage prennent une `--seed` : mêmes images à chaque appel, sur tout device.
- Chaque run enregistre sa config, son commit git et son environnement (Python, torch, matériel).
- Un Mac et un GPU ne donnent pas les mêmes poids : la baseline de référence est entraînée sur Juliet.

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

## Contraintes du sujet (rappel)

- Ne pas modifier l'architecture du générateur (couches, largeurs, dimensions d'entrée et de sortie).
- `python generate.py --batch_size 256` charge `checkpoints/` et produit **exactement 10 000** images 28×28 dans `samples/`.
- Toutes les dépendances sont dans `requirements.txt`. Le correcteur lance `pip install -r requirements.txt`.
- Pousser le minimum de modèles dans `checkpoints/`.
- Rendus :
  - `report.pdf` (5 pages) : 11/11 à 23h59 ;
  - `slides.pdf` : 4/11 ou 11/11 selon le groupe ;
  - code final : 12/11 à 05h00.

## Accès à MesoNet (Juliet)

1. Créer un compte sur https://iam.mesonet.fr (eduGAIN, Dauphine), puis compléter son profil sur https://acces.mesonet.fr.
2. Demander à être ajouté au projet IASD (alexandre.verine@dauphine.psl.eu).
3. Ajouter sa clé SSH publique : https://acces.mesonet.fr/gramc-meso/ → Utilisateur → Clés ssh. L'associer ensuite à Juliet dans la liste des projets.
4. `ssh <login>@juliet.mesonet.fr`. Documentation : https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion
