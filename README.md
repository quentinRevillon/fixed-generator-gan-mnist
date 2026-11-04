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
| Le volet Données | `rapport-donnée-illustré.md` | `nico/preprocessing` |
| Le volet Entraînement | `training-r3gan.md` | `noe/training` |
| Les choix d'organisation et leurs sources | [architecture-choix.md](architecture-choix.md) | toutes |
| Les slides | `slides/comparaison.tex`, `slides/slides.pdf` | `quentin/comparaison` |
| La comparaison sur protocole commun | `results/comparison_local.csv` | `quentin/comparaison` |

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
