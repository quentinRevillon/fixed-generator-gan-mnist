# Choix d'architecture du projet

Ce document explique **comment le repo est organisé et pourquoi**, pour que les 3 membres du groupe travaillent de la même façon et puissent défendre chaque choix à l'oral.

## 1. Organisation du repo

```
configs/baseline.yaml      la baseline commune (une config = une expérience)
model.py                   Generator / Discriminator — imposés, NON modifiés
utils.py                   pas d'entraînement D/G, config, seed, checkpoints
train.py                   entraînement : train(cfg) + CLI --config / --resume
evaluate.py                FID, precision / recall, grille, histogramme des classes
tune.py                    recherche d'hyperparamètres (Optuna)
generate.py                10 000 images dans samples/ (interface imposée)
scripts/{train,tune,generate}.sh   lancement Slurm sur Juliet
checkpoints/G.pth          modèle rendu (lu par generate.py)
checkpoints/mnist_classifier.pth   extracteur de features partagé pour l'évaluation
runs/<name>/               sorties d'un run (ignoré par git)
results.csv                une ligne par évaluation (versionné)
```

## 2. Config YAML plutôt que des options en ligne de commande

- Une expérience = un fichier versionné : on la relit, on la compare (`git diff configs/a.yaml configs/b.yaml`) et on la relance à l'identique.
- Les commandes restent courtes : `python train.py --config configs/xxx.yaml`.
- À trois, chacun crée ses fichiers (`configs/data_diffaug.yaml`, ...) sans toucher à ceux des autres.
- En YAML, le séparateur décimal est le **point** : `0.0002` est un nombre, `0,0002` serait lu comme du texte.

## 3. La baseline

| Paramètre | Valeur | Raison |
|---|---|---|
| Époques | 50 | Assez pour converger sur MNIST, environ 20 min sur un Mac M-series. |
| lr | 0.0002 | Valeur standard des GAN (DCGAN), déjà celle du code fourni. |
| batch_size | 100 | 600 batchs par époque, un compte rond. |
| Optimiseur | Adam (0.9, 0.999) | Celui du code fourni. β1 = 0.5 est un candidat testé par Optuna. |
| Seed | 0 | Reproductibilité. |
| Validation croisée | aucune | Un GAN ne se valide pas par validation croisée : on l'évalue par FID / P / R sur le MNIST test, jamais vu à l'entraînement. |

Tout le monde compare ses méthodes **à cette baseline**, en ne changeant qu'une chose à la fois (slide 84).

## 4. Seed et reproductibilité

- `set_seed` fixe `random`, `numpy` et `torch`. Le DataLoader a son propre générateur, initialisé avec la seed, ce qui fixe l'ordre des batchs.
- La copie `runs/<name>/config.yaml` contient un champ `git_commit` : l'identifiant de la version du code qui a produit le run. On sait toujours quel code a donné quel résultat.
- Sur GPU, certaines opérations restent légèrement non déterministes. C'est pourquoi on compare idéalement 2 ou 3 seeds avant de conclure qu'une méthode marche mieux.

## 5. Dossier de run et checkpoints

Un run écrit dans `runs/<name>/` (`name` et `output.dir` sont dans le YAML) :
- `config.yaml` : la config exacte + `git_commit` ;
- `events.out.tfevents.*` : les logs TensorBoard ;
- `G.pth` : le générateur seul, au même format que `checkpoints/G.pth` ;
- `ckpt/last.pth` à chaque époque, et `ckpt/epoch_XXX.pth` tous les `save_every`. Ils contiennent G, D, les deux optimiseurs et l'époque ;
- `grid.png` : 100 images générées en fin de run.

**Pourquoi :**
- **Rien n'est jamais écrasé :** relancer un run qui existe déjà s'arrête avec un message.
- **Reprise possible :** `--resume` repart de `last.pth`, utile quand Slurm coupe un job à sa limite de temps.
- **Promotion explicite :** le modèle rendu reste `checkpoints/G.pth`. On le remplace volontairement par le meilleur run avec `cp runs/<name>/G.pth checkpoints/G.pth`.

## 6. TensorBoard plutôt que Weights & Biases

- Pas de compte ni de clé API, et ça fonctionne hors ligne, ce qui compte sur Juliet où les nœuds n'ont pas forcément Internet.
- Un run est un simple dossier : on partage `runs/` (sans les `.pth`) et `tensorboard --logdir runs` affiche les runs de tout le monde.
- **Ce qui est enregistré :**
  - les pertes de D et G, D(x) et D(G(z)) ;
  - une grille de 64 images à **bruit z fixe** à chaque époque, pour voir le même tirage évoluer ;
  - FID, precision, recall et l'histogramme des classes tous les `eval.every`.

## 7. Post-traitement des images : `(x + 1) / 2`

- Les données d'entraînement sont normalisées dans [-1, 1] (`Normalize(0.5, 0.5)`), d'où le `tanh` en sortie du générateur.
- Le code fourni enregistrait directement cette sortie : `save_image` coupe les valeurs négatives à 0, ce qui fausse les niveaux de gris.
- `generate.py` applique donc l'inverse de la normalisation avant d'enregistrer. Vérification : l'intensité moyenne des images générées est passée à 0,131, celle du vrai MNIST.
- **C'est autorisé :** la consigne (slides 74-75) interdit de changer **l'architecture** du générateur, et `model.py` n'est pas modifié. Il s'agit d'un post-traitement de la sortie, comme le prévoit le slide 74 (« investigate preprocessing, objectives and sampling »).

## 8. Évaluation (`evaluate.py`)

### Pourquoi un « extracteur de features »
FID, precision et recall ne comparent pas des pixels mais des **représentations** d'images. On entraîne une fois un petit CNN à reconnaître les chiffres (98,7 % de précision sur le test) et on garde son avant-dernière couche : un vecteur de 128 nombres par image. Deux façons d'écrire un 7 ont ainsi des vecteurs proches, même si leurs pixels diffèrent.

- **Pourquoi pas Inception (le FID standard) :** il faut télécharger environ 100 Mo de poids, et ce réseau a été entraîné sur des photos couleur, pas sur des chiffres.
- **Pourquoi il est versionné** (`checkpoints/mnist_classifier.pth`) : tout le groupe utilise le **même** extracteur, donc vos chiffres sont comparables entre eux.
- **Limite :** ces chiffres ne sont pas comparables en valeur absolue avec ceux de l'évaluation officielle. Seules les comparaisons internes ont du sens.
- On prend les features **avant** la dernière ReLU : après la ReLU, certaines dimensions valent toujours 0, ce qui rend la covariance singulière et le FID instable.

**Référence :** les 10 000 images du **MNIST test**, jamais vues à l'entraînement, comparées à 10 000 images générées.

### FID (Fréchet Inception Distance)
On modélise les features réelles et générées par deux gaussiennes de moyennes μ et de covariances Σ, et on calcule leur distance :

`FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})`

- **Intuition :** le premier terme compare les « centres » des deux nuages de points, le second compare leurs formes et leurs étalements.
- Plus c'est **bas**, mieux c'est. Le FID mélange qualité et diversité en un seul nombre.

### Precision et recall (k plus proches voisins, Kynkäänniemi et al. 2019)
1. Pour chaque image réelle `r_i`, on calcule `ρ_i` = la distance à son k-ième plus proche voisin réel (k = 3). La réunion des boules `B(r_i, ρ_i)` approxime la « zone » occupée par les vraies données.
2. **Precision** = la proportion d'images générées qui tombent dans au moins une boule réelle. Elle mesure la **qualité** : les images générées ressemblent-elles à de vraies images ?
3. **Recall** = même calcul dans l'autre sens : la proportion d'images réelles qui tombent dans la zone des images générées. Elle mesure la **diversité / couverture** : le générateur couvre-t-il toutes les variantes ?

Ce couple sépare ce que le FID mélange. Par exemple, un *mode collapse* donne une bonne precision mais un recall très faible.

### Inspection visuelle et histogramme des classes
- `grid.png` : 100 images générées, pour les slides et pour repérer les défauts à l'œil.
- **Histogramme des chiffres prédits** par le classifieur sur les images générées : il montre directement quels chiffres le GAN sur-produit ou oublie. C'est un indicateur clé pour la partie Données (sélection, équilibrage).

### Valeurs de repère (mesurées)
| Comparaison | FID | Precision | Recall |
|---|---|---|---|
| MNIST train vs MNIST test (**plafond réaliste**) | 5,1 | 0,87 | 0,87 |
| `checkpoints/G.pth` (baseline actuelle) | 99,3 | 0,80 | 0,75 |

- Même deux échantillons de vraies données n'atteignent pas P = R = 1 avec k = 3 : 0,87 est le maximum atteignable en pratique.
- La baseline actuelle génère très peu de « 2 » (3,5 % au lieu d'environ 10 %).

## 9. Optuna (`tune.py`)

- **Espace de recherche :**
  - `lr` entre 1e-5 et 1e-3, en échelle log ;
  - `beta1` dans {0, 0.5, 0.9} ;
  - `batch_size` dans {64, 100, 128}.
- **Budget fixe** (`--epochs`, 20 par défaut) pour tous les essais : on compare à coût égal.
- **Objectif :** minimiser le FID. Precision et recall sont enregistrés pour chaque essai, pour discuter du compromis.
- **Pruner** (`MedianPruner`) : un essai dont le FID intermédiaire est moins bon que la médiane des autres est arrêté tôt, ce qui économise du calcul.
- **Stockage :** l'étude est enregistrée dans `runs/optuna.db`. Relancer la même commande reprend l'étude, et chaque essai est un run TensorBoard (`runs/<study>/trial_XXX/`).
- Les meilleurs hyperparamètres trouvés deviennent une nouvelle config. Ils ne s'appliquent jamais automatiquement à la baseline.

## 10. Conventions d'équipe

- **Une branche par personne / volet** (`nico/preprocessing`, `noe/training`, `quentin/inference`). On fusionne dans `main` ce qui est validé.
- **Une config par expérience**, nommée explicitement (`data_diffaug_p0.3`, `train_wgangp`, ...), avec la seed dans le nom si on en compare plusieurs.
- **Un seul changement à la fois** par rapport à la baseline, puis les combinaisons utiles.
- **`results.csv`** : chaque évaluation ajoute une ligne (date, run, FID, P, R, répartition des classes). On le commite : c'est le tableau du rapport.
- On évalue le rendu final exactement comme le correcteur : `python generate.py` puis `python evaluate.py --samples samples`.
