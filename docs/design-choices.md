# Choix d'architecture du projet

Ce document explique **comment le repo est organisé et pourquoi**, pour que les 3 membres du groupe travaillent de la même façon et puissent défendre chaque choix à l'oral.

Il reprend le document de Nicodème. Chaque choix indique maintenant sa source, avec l'une de ces trois étiquettes :

| Étiquette | Sens |
|---|---|
| **[Papier]** | un article le justifie ; le passage est cité |
| **[Cours]** | le justificatif vient des slides ou de la plateforme du prof |
| **[Sans papier]** | choix pratique, aucun article ne le justifie : c'est assumé, on ne force pas de citation |

Les citations ont été relevées dans le texte des papiers. Références complètes au §13. Les points où une source va **contre** notre choix sont résumés au §11.

## 1. Organisation du repo

```
configs/baseline_200ep_prof.yaml   la baseline de référence (une config = une expérience)
configs/sampling/*.yaml    une config par méthode d'échantillonnage
model.py                   Generator / Discriminator — fournis, NON modifiés
utils.py                   pas d'entraînement D/G, config, seed, déterminisme, checkpoints
data.py                    split figé train / val / test
train.py                   entraînement : train(cfg) + CLI --config / --resume
evaluate.py                FID, precision / recall, classes, tableaux de résultats
sampling.py                inférence : plain (quentin/baseline) ; truncation, DRS, OBRS (quentin/inference)
keep_training.py, sweep.py étape KeepTraining du DRS, variation d'un paramètre (quentin/inference)
tune.py                    recherche d'hyperparamètres (Optuna)
generate.py                10 000 images dans samples/ (interface imposée)
scripts/*.sh               lancement Slurm sur Juliet
checkpoints/               modèle rendu : G.pth, D.pth, sampling.yaml (+ mnist_classifier.pth)
runs/<name>/               sorties d'un run (dossier léger versionné, voir §5)
results/                   une ligne par évaluation : results.csv (petit CNN), results_prof.csv (Inception, VGG)
figures/                   figures par volet et scripts make_*_figures.py
tests/                     tests unitaires : python -m pytest tests -q
```

- **Structure à plat, sans package.** [Sans papier] C'est le plus simple pour une dizaine de fichiers à trois.
- **Interface de `generate.py` conservée.** [Cours] Slide 85 : *« Keep the starter interface »*.

## 2. Config YAML plutôt que des options en ligne de commande

- Une expérience = un fichier versionné : on la relit, on la compare (`git diff configs/a.yaml configs/b.yaml`) et on la relance à l'identique.
- Les commandes restent courtes : `python train.py --config configs/xxx.yaml`.
- À trois, chacun crée ses fichiers (`configs/data_diffaug.yaml`, ...) sans toucher à ceux des autres.
- En YAML, le séparateur décimal est le **point** : `0.0002` est un nombre, `0,0002` serait lu comme du texte.

**Source.** [Cours] Cours Slurm : *« Save the final resolved settings with every run »* et *« Choose one convention and document it »*. Le format YAML lui-même est [Sans papier].

## 3. La baseline

| Paramètre | Valeur | Source et raison |
|---|---|---|
| Générateur | imposé | [Cours] Slide 74 : *« Do not change the generator architecture »*. |
| Objectif | BCE, G non saturant | [Papier] Goodfellow 2014 : *« Rather than training G to minimize log(1 - D(G(z))) we can train G to maximize log D(G(z)) »*. C'est aussi le code fourni. |
| lr | 0.0002 | [Papier] DCGAN : *« We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead »*. C'est aussi le code fourni. |
| Optimiseur | Adam (0.9, 0.999) | Code fourni. **Contredit par DCGAN** sur β1, voir §11. |
| batch_size | 100 | [Sans papier] 500 batchs par époque sur 50 000 images. DCGAN utilise 128, le code fourni 64. |
| Époques | 200 | [Sans papier] Choix mesuré, voir ci-dessous. |
| Seed | 0 | [Papier] Lucic 2018 : *« varying the random seed may influence on the results »*. |
| Validation | split fixe | Voir ci-dessous. |

**Le split train / val / test** (`data.py`) : les 60 000 images de MNIST train sont permutées une fois (seed 0).

| Split | Taille | Sert à |
|---|---|---|
| `train` | 50 000 | entraîner G et D |
| `val` | 10 000 | choisir : époque, hyperparamètres, K, ψ |
| `test` | 10 000 (MNIST test) | chiffres finaux uniquement |

- [Cours] Cours Slurm : *« Never tune on the test set »*, *« Keep the test set for the final evaluation »*, *« Best is selected on validation »*.
- [Papier] Pineau 2020 : la checklist de NeurIPS comprend des *« items related to test/validation/train splits, hyper-parameter ranges »*.
- [Sans papier] La taille 50k / 10k est un choix pratique. La perte de 10 000 images d'entraînement n'a pas été mesurée.

**Le nombre d'époques.** La baseline de 50 époques n'a pas convergé. Mesuré sur Juliet (`runs/baseline_v1`), FID sur `val` avec le petit CNN :

| Époque | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| FID | 2449 | 1821 | 892 | 459 | 323 |

Le run de référence va donc jusqu'à 200 époques (`configs/baseline_200ep_prof.yaml`). Ses résultats et le choix de l'époque sont dans `baseline.md`.

Tout le monde compare ses méthodes **à cette baseline**, en ne changeant qu'une chose à la fois. [Cours] Slide 84 : *« State a hypothesis and change one component at a time »*.

## 4. Seed et reproductibilité

- `set_seed` fixe `random`, `numpy` et `torch`. Le DataLoader a son propre générateur, initialisé avec la seed, ce qui fixe l'ordre des batchs.
- `make_deterministic()` demande à PyTorch des algorithmes déterministes. [Sans papier] Réglage technique.
- L'évaluation tourne dans `fork_rng` avec son propre générateur : évaluer ou non ne change pas les poids.
- `last.pth` contient les états RNG : `--resume` reprend à l'identique. [Cours] Cours Slurm : *« Resume needs model, optimizer, epoch and relevant random states »*.
- `runs/<name>/config.yaml` contient `git_commit` et `environment` (versions, matériel). [Cours] Cours Slurm : *« Keep the seed and software versions »*.

**Vérifié sur CPU, 2 époques :** avec ou sans évaluation, et avec ou sans reprise, `G.pth` est identique au bit près.

**Vérifié sur GPU, 200 époques :** deux runs sur deux nœuds A100 différents (`juliet3` et `juliet2`), avec des mesures toutes les 10 ou toutes les 2 époques, donnent des poids de G et de D identiques au bit près.

**Limites :**
- [Cours] Cours Slurm : *« The same seed is useful, but does not guarantee identical results on different hardware »*. D'où une seule baseline de référence, entraînée sur Juliet.
- [Papier] Pour conclure qu'une méthode est meilleure, on compare plusieurs seeds. Lucic 2018 : *« a given algorithm can achieve very different scores, when varying the architecture, hyperparameters, random initialization »*. Slide 84 : *« If possible, repeat runs to distinguish an improvement from seed variability »*.

## 5. Dossier de run et checkpoints

Un run écrit dans `runs/<name>/` (`name` et `output.dir` sont dans le YAML) :

| Fichier | Contenu | Versionné |
|---|---|---|
| `config.yaml` | config exacte, `git_commit`, `environment` | oui |
| `metrics.jsonl` | une ligne par époque : pertes, D(x), D(G(z)), FID / P / R | oui |
| `G.pth`, `D.pth` | dernière époque | oui |
| `G_best.pth`, `D_best.pth` | époque de meilleur FID sur `val` | oui |
| `grid.png` | 100 images de la dernière évaluation | oui |
| `ckpt/last.pth`, `ckpt/epoch_XXX.pth` | état complet pour `--resume` | non (34 Mo chacun) |
| `events.out.tfevents.*` | logs TensorBoard | non |

**Pourquoi :**
- **Rien n'est jamais écrasé :** relancer un run qui existe déjà s'arrête avec un message. [Cours] Cours Slurm : *« Use a new run directory for a new experiment »*.
- **Reprise possible :** utile quand Slurm coupe un job à sa limite de temps.
- **Meilleure époque :** [Cours] Cours Slurm, slide *« Checkpoints: last is not best »*.
- **D est versionné :** [Papier] DRS et OBRS ont besoin du discriminateur (§12).
- **Modèles dans Git :** [Cours] Le cours Slurm conseille de garder les gros checkpoints hors de Git, et le README du prof demande *« the minimal amount of models »*. On ne versionne donc que les fichiers légers (environ 6 Mo chacun). À limiter aux runs utiles.
- **Promotion explicite :** le modèle rendu reste dans `checkpoints/`. On y copie volontairement le meilleur couple G et D, et on écrit la méthode dans `checkpoints/sampling.yaml`.

## 6. TensorBoard plutôt que Weights & Biases

- [Sans papier] Pas de compte ni de clé API, et ça fonctionne hors ligne. Nicodème a essayé W&B et ça n'a pas marché.
- [Cours] Le cours Slurm présente les deux outils et ajoute : *« Keep local JSONL results even if a tracking service is unavailable »*. D'où `metrics.jsonl`.
- **Ce qui est enregistré :**
  - les pertes de D et G, D(x) et D(G(z)) ;
  - une grille de 64 images à **bruit z fixe** à chaque époque, pour voir le même tirage évoluer ;
  - FID, precision et recall tous les `eval.every`.

## 7. Post-traitement des images : `(x + 1) / 2`

- Les données d'entraînement sont dans [-1, 1], d'où le `tanh` en sortie du générateur. [Papier] DCGAN : *« No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1] »*.
- Le code fourni enregistrait directement cette sortie : `save_image` coupe les valeurs négatives à 0, ce qui fausse les niveaux de gris.
- `generate.py` applique donc l'inverse de la normalisation avant d'enregistrer. [Sans papier] C'est une correction de code. Vérification faite par Nicodème : l'intensité moyenne des images générées est passée à 0,131, celle du vrai MNIST.
- **C'est autorisé :** la consigne (slides 74-75) interdit de changer **l'architecture** du générateur. `model.py` n'est pas modifié.

## 8. Évaluation (`evaluate.py`)

### Deux mesures à ne pas confondre

| | Classement du prof | `--metric prof` | `--metric cnn` |
|---|---|---|---|
| FID | Inception, toutes les images | Inception, 10 000 images | petit CNN MNIST, 128 dim. |
| Précision / rappel | VGG, k = 5, 9 984 images | VGG16, k = 5, 9 984 images | même CNN, k = 3, 10 000 images |
| Référence réelle | non précisée | `val`, ou `test` pour les chiffres finaux | idem |

[Cours] Page de résultats du prof : *« Historical FID uses Inception features on all images; historical P/R uses VGG features with k=5 and 9,984 images (complete batches of 256) »*. Le prétraitement, les poids et la version de VGG ne sont pas précisés.

**Le même modèle donne des chiffres très différents :** `checkpoints/G.pth` obtient FID 44,0 / P 0,705 / R 0,502 au classement, FID 43,2 / P 0,389 / R 0,429 avec `--metric prof` contre `val`, et FID 97,6 / P 0,834 / R 0,744 avec `--metric cnn` contre `val`. On ne compare jamais deux mesures différentes.

**Notre FID Inception est proche du classement, notre précision VGG non** (détails dans `baseline.md`). Les précisions et rappels ne servent donc qu'à comparer nos configurations entre elles.

**Les deux mesures ne choisissent pas la même époque :** le petit CNN est très sensible au mélange des chiffres, Inception beaucoup moins. `baseline.md` le montre. On garde les deux : Inception et VGG pour décider sur la même mesure que le classement, le CNN pour voir l'équilibre des chiffres.

### Pourquoi un petit CNN entraîné sur MNIST

FID, precision et recall ne comparent pas des pixels mais des **représentations** d'images. On a entraîné une fois un petit CNN à reconnaître les chiffres (98,7 % de précision sur le test), et on garde un vecteur de 128 nombres par image.

**[Papier] Issenhuth 2021 fait la même chose sur MNIST pour comparer des méthodes de post-traitement :**
- Section 4.1 : *« φ is a pre-softmax layer of a supervised classifier. On MNIST and F-MNIST, the classifier is pre-trained on the given dataset. On CelebA and LSUN Church, we use VGG-16 pre-trained on ImageNet. »*
- Table 3 : *« FID was computed using the same dataset-specific classifier used for the Precision/Recall metric. »*

**Limites de cet appui :**
- Pour le FID, le papier se contredit : son annexe C écrit *« For FID, we use the standard protocol with n = 50000 points and Inception Net »*. Seule la justification pour la précision et le rappel est nette.
- Le papier utilise ce classifieur sans argumenter ce choix, et ne décrit ni son architecture ni la couche exacte. Notre CNN précis reste un choix de Nicodème.
- Protocole différent : ils utilisent 2048 images et 10 évaluations, nous 10 000 images et 3 seeds.

**Autres points :**
- **Pourquoi il est versionné** (`checkpoints/mnist_classifier.pth`) : tout le groupe utilise le **même** extracteur.
- **Il a vu les 60 000 images de MNIST train, validation comprise.** [Sans papier] Acceptable : il définit seulement l'espace de features, et aucun choix n'est fait avec lui.
- **Features avant la dernière ReLU :** [Sans papier] Après la ReLU, certaines dimensions valent toujours 0, ce qui rend la covariance singulière et le FID instable.

### FID (Fréchet Inception Distance)

`FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})`

- **Intuition :** le premier terme compare les « centres » des deux nuages de points, le second compare leurs formes et leurs étalements.
- [Papier] Heusel 2017 calcule ces statistiques avec Inception-v3 : *« we use the last pooling layer as coding layer »*. Notre formule est la même ; seul le réseau change.
- Plus c'est **bas**, mieux c'est. [Cours] Slide 9 : *« One number mixes quality and diversity »*.

**Nombre d'images :**
- [Cours] La plateforme impose 10 000 images.
- [Papier] Heusel utilise 50 000 images générées. Lucic 2018 note que le FID a *« a slight bias, but low variance on samples of size 10000 »*.
- [Papier] **Limite :** Chong 2020 montre que le FID à échantillon fini est biaisé selon le générateur, et que *« This effect cannot be fixed by evaluating at a fixed number of samples »*. Nos comparaisons de FID restent donc approximatives, et un petit écart ne prouve rien.

### Precision et recall (k plus proches voisins)

1. Pour chaque image réelle `r_i`, on calcule `ρ_i` = la distance à son k-ième plus proche voisin réel. La réunion des boules `B(r_i, ρ_i)` approxime la « zone » occupée par les vraies données.
2. **Precision** = la proportion d'images générées qui tombent dans au moins une boule réelle : la **qualité**.
3. **Recall** = même calcul dans l'autre sens : la **diversité / couverture**.

- [Papier] C'est la méthode de Kynkäänniemi 2019. Sur k : *« Tests with various datasets and GANs showed that k = 3 is a robust choice »*.
- [Papier] Eux utilisent VGG-16, *« after the second fully connected layer »*, et 50 000 images. Ils notent aussi que *« Inception-v3 features lead to very similar results »*. Ils ne testent pas de petit CNN.
- [Cours] Slide 86 : *« The theoretical PR curve and the finite-sample image estimator are different objects »*. On teste des tendances, pas la formule théorique.

### Inspection visuelle et histogramme des classes

- [Cours] Slide 33 : la plateforme retient *« the visual inspection of the generated samples »*. D'où `grid.png`.
- [Cours] L'histogramme des chiffres prédits montre quels chiffres le GAN sur-produit ou oublie. Slide 83 : *« Manifold uniformity is not the same as semantic class balance »*. Il n'existe qu'avec notre classifieur MNIST, ce qui justifie de le garder.

### Valeurs de repère (notre mesure, mesurées)

| Comparaison | FID | Precision | Recall |
|---|---|---|---|
| 10 000 images du split train vs `val` (**plafond**) | 1,7 | 0,90 | 0,90 |
| 10 000 images du split train vs `test` (**plafond**) | 3,1 | 0,87 | 0,89 |
| `checkpoints/G.pth`, plain, vs `val` (3 seeds) | 97,6 ± 3,6 | 0,834 | 0,744 |

- Même deux échantillons de vraies données n'atteignent pas P = R = 1.
- `val` est plus proche du train que `test` : on ne compare jamais un chiffre « val » à un chiffre « test ».
- Ce modèle génère très peu de « 2 » (3,4 % au lieu d'environ 10 %).

## 9. Optuna (`tune.py`)

**Optuna n'est pas demandé par les consignes A2.** Il n'apparaît que dans le cours Slurm, comme outil facultatif.

- **Espace de recherche :** `lr` entre 1e-5 et 1e-3 en échelle log, `beta1` dans {0, 0.5, 0.9}, `batch_size` dans {64, 100, 128}. [Sans papier] Le 0.5 vient de DCGAN.
- **Budget fixe** pour tous les essais. [Papier] Lucic 2018 : *« improvements can arise from a higher computational budget and tuning more than fundamental algorithmic changes »*. [Cours] Slide 81 : *« Compare at a stated compute budget »*.
- **Objectif :** minimiser le FID sur `val`. [Cours] Cours Slurm : *« Never tune on the test set »*.
- **Outil :** [Papier] Akiba 2019.
- **Stockage :** `runs/optuna.db`. Les meilleurs hyperparamètres deviennent une nouvelle config, jamais automatiquement la baseline.

**Limites :**
- [Cours] Cours Slurm : *« a single winner is not evidence of a robust gain »*.
- **Le budget par défaut est trop court :** `--epochs 20`, alors que la baseline a encore un FID de 1821 à l'époque 20. À augmenter avant toute utilisation.
- **Le pruner n'est pas le meilleur selon le papier :** `tune.py` utilise `MedianPruner`, alors qu'Akiba 2019 montre *« the superiority of ASHA pruning over median pruning »*.

## 10. Conventions d'équipe

- **Une branche par personne / volet** (`nico/preprocessing`, `noe/training`, `quentin/inference`), et une branche pour la baseline commune (`quentin/baseline`). [Cours] Slide 76 : *« Share a baseline, checkpoints, interfaces and evaluation protocol »*.
- **Une config par expérience**, nommée explicitement (`data_diffaug_p0.3`, `train_wgangp`, ...), avec la seed dans le nom si on en compare plusieurs.
- **Un seul changement à la fois**, puis les combinaisons utiles. [Cours] Slide 76 : *« Compare each intervention separately, then useful combinations »*.
- **`results/results.csv` et `results/results_prof.csv`** : chaque `evaluate.py --run` ou `sweep.py` ajoute une ligne par seed (run, méthode, paramètres, FID, P, R, coût, classes) au tableau de sa mesure. On les commite.
- On évalue le rendu final comme le correcteur : `python generate.py` puis `python evaluate.py --samples samples`.

## 11. Là où une source va contre notre choix

| Notre choix | Ce que dit la source | Statut |
|---|---|---|
| β1 = 0.9 | DCGAN : *« suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training »* (GAN convolutionnel, pas MLP) | gardé car c'est le code fourni ; à tester |
| FID et P/R sur petit CNN | la plateforme utilise Inception et VGG | résolu : les deux mesures existent (§8) |
| k = 3 | Kynkäänniemi 2019 recommande k = 3, la plateforme utilise k = 5 | k = 3 pour le CNN, k = 5 pour `--metric prof` |
| Référence `val` / `test` | Heusel calcule les statistiques réelles sur *« all images from the training dataset »* ; la référence de la plateforme est inconnue | à demander au prof |
| D utilisé sans recalibration | Azadi 2019 continue l'entraînement de D *« using early stopping on the validation set »* ; Turner 2019 recalibre sur *« a held out calibration set (e.g., 10% »* | KeepTraining implémenté et mesuré : voir `inference.md` |
| Comparaisons de FID à 10 000 images | Chong 2020 : le biais dépend du générateur | limite assumée |
| `MedianPruner` | Akiba 2019 : ASHA fait mieux | à changer si on garde Optuna |

## 12. Volet Inférence (`sampling.py`)

Détails et résultats : `inference.md`, sur la branche `quentin/inference`.

**Consigne.** [Cours] Slide 82 : *« Keep the trained generator. Report acceptance rates, total generation time and the full evaluation batch »*. Aucune méthode ne réentraîne G.

| Choix | Source |
|---|---|
| Utiliser D comme rapport de densités | [Papier] Goodfellow 2014 : *« For G fixed, the optimal discriminator »* vaut `p_data / (p_data + p_g)`. Che 2020 l'écrit `e^{d(x)} ≈ p_d / p_g`, avec d le logit de D. |
| Logit avant la sigmoïde | [Papier] Azadi 2019 et Che 2020 raisonnent sur le logit. La stabilité numérique est [Sans papier]. |
| **DRS, méthode implémentée comme dans le papier** | [Papier] Azadi 2019, Figure 1 et équation 8 : KeepTraining, 10 000 images pour estimer M, γ par percentile dans chaque batch. ε et les réglages de KeepTraining sont [Sans papier]. |
| OBRS, comparaison | [Papier] Verine 2024, Thm 3.1 (Thm 3.2 des slides) : acceptation `min((p/p̂)(c_K/M), 1)`, optimale à budget K pour toute f-divergence. Thm 3.3 (Prop. 3.3 des slides) : effet sur précision et rappel. |
| DRS réglé par K, comparaison | [Papier] Azadi 2019, slides Def. 3.1. γ calibré pour le même budget K : [Sans papier], pour comparer à coût égal (slide 81). |
| Troncature, comparaison | [Papier] Brock 2019 : *« truncating a z vector by resampling »*. Sans régularisation orthogonale, *« only 16% of models are amenable to truncation »*. |
| Hasard sur CPU avec un générateur dédié | [Sans papier] Même seed, mêmes images. |

**Brancher une nouvelle méthode :** écrire `sample_xxx(G, D, n, rng, batch_size, **params)`, l'ajouter à `SAMPLERS`, créer `configs/sampling/xxx.yaml`.

## 13. Références

**Inférence :**
- Azadi et al. (2019). *Discriminator Rejection Sampling*. ICLR.
- Brock et al. (2019). *Large Scale GAN Training for High Fidelity Natural Image Synthesis*. ICLR.
- Che et al. (2020). *Your GAN is Secretly an Energy-based Model and You Should Use Discriminator Driven Latent Sampling*. arXiv:2003.06060.
- Issenhuth et al. (2021). *Latent reweighting, an almost free improvement for GANs*.
- Turner et al. (2019). *Metropolis-Hastings Generative Adversarial Networks*.
- Verine et al. (2024). *Optimal Budgeted Rejection Sampling for Generative Models*. arXiv:2311.00460.

**Autres :**
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. arXiv:1907.10902.
- Chong & Forsyth (2020). *Effectively Unbiased FID and Inception Score and where to find them*. arXiv:1911.07023.
- Goodfellow et al. (2014). *Generative Adversarial Nets*. arXiv:1406.2661.
- Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. arXiv:1706.08500.
- Kynkäänniemi et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models*. arXiv:1904.06991.
- Lucic et al. (2018). *Are GANs Created Equal? A Large-Scale Study*. arXiv:1711.10337.
- Pineau et al. (2020). *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*. arXiv:2003.12206.
- Radford et al. (2016), « DCGAN ». *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434.

**Cours :** slides « Evaluating and tuning Precision and Recall » (A2), slides « From a notebook to experiments on a cluster » (Slurm), [page de résultats de la plateforme](https://www.lamsade.dauphine.fr/~testplatform/prds-a2/results/last.html).
