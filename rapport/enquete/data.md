# Volet Données (Nicodème) : attendu vs observé, critique, corrections

Date : 2026-11-08. Lecture seule du dépôt, du worktree et des branches. Tous les calculs sont refaits localement.

**Sources de données (abréviations utilisées partout) :**
- `runs.csv`, `summary.csv`, `saturation.csv`, `real_subsets.csv`, `per_class_fid.csv`, `class_gaps.csv` : `figures/data/*.csv` sur `origin/nico/preprocessing` (commit 56531a4, 2026-11-04). Ils sont **identiques octet par octet** à ceux du worktree `wt-comparaison` (vérifié avec `cmp`). Seuls `preprocessing.py`, `train.py` et `evaluate.py` diffèrent : le worktree contient la version « protocole commun » (split 50k, Inception), qui n'a pas servi aux runs Données.
- `cmp.csv` : `wt-comparaison/results/comparison_local.csv` (protocole commun, 3 seeds).
- `rapport` : `rapport-donnée-illustré.md` sur `origin/nico/preprocessing`. `tex` : `slides/comparaison.tex` sur main. `oral` : `slides/oral-data.md` sur main.
- Scripts de vérification (scratchpad, `scripts/data/`) : `stats.py` (sortie `stats_out.txt`), `seed_table.py` (`seed_table_out.txt`), `noise_mechanics.py` (`noise_mechanics_out.txt`).

**Notions utilisées :**
- **FID** (Fréchet Inception Distance) : distance entre les nuages de features des vraies images et des images générées. Plus bas = mieux. Ici, le « CNN FID » utilise les features (128 dim) du petit classifieur MNIST `checkpoints/mnist_classifier.pth`.
- **Precision** : part des images générées qui tombent dans la zone des vraies images (la qualité). **Recall** : part des vraies images couvertes par les images générées (la diversité). Ici, on les calcule par k plus proches voisins, avec k = 3.
- **Run effondré** (*collapsed*) : le générateur sort la même tache pour tout z. Critère du code : recall final < 0,01 (`make_data_figures.py:83`).
- **Taux de rétention** : part des images d'entraînement gardée par la sélection.
- **Test exact de Fisher** : compare deux proportions (par ex. 3/8 contre 0/3) sans approximation. **Test de Welch** : compare deux moyennes sans supposer des variances égales. **p** : probabilité d'observer un écart au moins aussi grand si en réalité il n'y a aucun effet. Avec p > 0,05, on ne peut pas conclure.
- **Biais du survivant** : on ne décrit que les runs qui ont réussi, en oubliant ceux qui ont échoué.
- **Circularité** : on mesure le résultat avec le même outil qui a servi à le fabriquer.

---

## 1. Attendu (littérature)

### 1.1 Sélection d'instances (DeVries, Drozdzal & Taylor, NeurIPS 2020)

**Principe :** ajuster une gaussienne par classe dans un espace de features pré-entraîné (Inception-v3, ImageNet), puis ne garder que les images de plus forte vraisemblance.

| Réglage du papier | FID | P | R | Source |
|---|---|---|---|---|
| SAGAN 64×64, demi-canaux, 200k it., 100 % | 21,4 | 0,66 | 0,62 | Table 1 |
| idem, **50 % au hasard** (contrôle) | 22,8 | 0,65 | 0,62 | Table 1 |
| idem, 50 % gaussienne Inception, par classe | **12,6** | 0,77 | 0,59 | Table 1 |
| idem, 50 %, Inception **non pré-entraîné** | 21,9 | 0,66 | 0,61 | Table 1 |
| SAGAN 64×64, 500k it. : 100 / 90 / 50 / 40 / 10 % | 17,23 / 15,85 / 9,63 / **9,07** / 19,24 | 0,68 / 0,70 / 0,79 / 0,81 / 0,79 | 0,66 / 0,66 / 0,62 / 0,60 / 0,33 | Table 6 |
| BigGAN 128×128, demi-capacité, 50 % | 11,5 → 9,6 | 0,76 → 0,88 | 0,66 → 0,50 | Table 2 |
| BigGAN 256×256, demi-capacité, 50 % | 9,8 → **10,6 (pire)** | 0,86 → 0,91 | 0,70 → 0,52 | Table 3 |

**Conditions du gain (vérifié dans le texte) :**
- ImageNet : 1 000 classes, avec beaucoup d'images atypiques ou ratées.
- Générateur **conditionnel** (SAGAN, BigGAN), avec une capacité réduite volontairement (§4.3, §4.5) : l'argument est de « réallouer la capacité » vers les zones denses.
- Un espace de features **aligné sur le domaine** : sans pré-entraînement, aucun gain (21,9 contre 21,4).
- Le gain ne vient pas du fait d'avoir moins d'images : le contrôle à 50 % au hasard ne gagne rien (22,8).
- Precision ↑, recall ↓ légèrement : −0,04 à 50 % en 64×64. Le FID s'améliore jusqu'à environ 70 % retirés (§4.4).

**Mises en garde du papier :**
- Le FID est calculé avec le **même Inception** que celui de la sélection, ce qui « peut donner un avantage non négligeable » (§4.3, fin).
- Avec la sélection, **D sur-apprend plus vite** et les modèles trop gros « s'effondrent tôt ». Remèdes proposés : réduire le batch et la capacité, ou ajouter une régularisation (« R1 », « augmentation »), non testée (Annexe F).

### 1.2 Bruit d'instance (Sønderby et al., ICLR 2017 ; Arjovsky & Bottou, ICLR 2017 ; Roth et al., 2017)

- **Sønderby et al. (arXiv 1610.04490), §3.2.1 et Annexe C :**
  - Motivation : les distributions réelle et générée sont concentrées sur des variétés de faible dimension dont les supports ne se recouvrent pas. Il existe alors beaucoup de D parfaits, et la théorie de convergence ne s'applique plus.
  - Remède : ajouter du bruit aux deux types d'images. On minimise alors KL[p_σ∗q ‖ p_σ∗p_Y].
  - Réglage, Annexe D : « Gaussian instance noise linearly annealed from an initial standard deviation of 0.1 to 0. We were unable to stable train models without this extra regularization. » La plage des pixels et la durée de la décroissance ne sont pas précisées.
  - **Aucun chiffre de gain.** C'est un argument de stabilité seulement, sur de la super-résolution ImageNet.
  - Les auteurs écrivent : « We have not tested the instance noise in the generative modelling application » (Annexe C).
  - Ils proposent aussi une décroissance **adaptative**, où σ est ajusté pour garder constante la perte du D optimal. Elle n'est pas testée chez nous.
- **Arjovsky & Bottou (arXiv 1701.04862) :**
  - Th. 2.1–2.2 : si les supports sont disjoints, ou sur des variétés qui ne se recouvrent pas, il existe un D parfait de gradient nul.
  - Th. 2.4 : le gradient de G disparaît avec la perte originale. Th. 2.6 : la variante −log D a des mises à jour instables.
  - §3, Th. 3.1–3.3 : ajouter un bruit continu aux entrées de D rend la JSD continue. W(P_r, P_g) ≤ 2V^{1/2} + 2C·√JSD(P_r+ε ‖ P_g+ε), où V est la variance du bruit, donc **le bruit doit décroître**.
  - **Pas d'expérience chiffrée sur le bruit.**
- **Roth, Lucchi, Nowozin & Hofmann (arXiv 1705.09367), §2 et §3 :**
  - Entraîner avec du bruit revient, au premier ordre, à **pénaliser la norme du gradient de D**.
  - Ajouter du bruit explicite en grande dimension « introduces significant variance » ; la pénalité analytique fait mieux (§5.4).
- **Mescheder et al. (ICML 2018), résumé :** « GAN training with instance noise or zero-centered gradient penalties converges ». La pénalité R1 est l'équivalent déterministe du bruit d'instance.

**Attendu chiffré :** aucun gain de FID n'est annoncé. On attend surtout **moins d'instabilité** (D moins sûr de lui) et un σ assez grand pour que les distributions se recouvrent. **Sur une perte déjà régularisée par R1, le bruit devrait être redondant.**

---

## 2. Observé chez nous (recalculé)

### 2.1 Protocole des runs Données (vérifié dans le code)

- Code : `train.py`, `preprocessing.py` et `configs/data/*.yaml` sur `nico/preprocessing`.
- Données : les **60 000** images de MNIST train (pas le split commun 50k/10k). Évaluation contre **MNIST test**, en CNN FID seulement, avec le classifieur entraîné sur ces mêmes 60k images.
- Perte : BCE de la baseline (`nn.BCELoss` sur la sortie sigmoïde de D, `train.py:58`). Pas d'EMA.
- Budget : 120 000 mises à jour, lues à **60 000** (évaluation la plus proche, `make_data_figures.py:77`). Les variantes sélectionnées tournent plus d'époques (400 pour 50 %).
- Bruit : `D(x + σ_t ε)` sur les vraies **et** les fausses images (`InstanceNoise`, utilisé dans `D_train` et `G_train`). σ_t = σ0·max(0, 1 − (époque−1)/100) : un escalier par époque, nul à partir de l'époque 101, soit 60k mises à jour. `sel_50_noise_01` utilise `anneal_epochs: 200`, donc lui aussi 60k mises à jour (vérifié dans le config).
- Sélection : `gaussian_score` = −distance de Mahalanobis² dans `embed(x, 'classifier')`, c'est-à-dire `predict(load_classifier(...))[0]`, soit la couche `fc` (128 dim, avant ReLU) de `checkpoints/mnist_classifier.pth` (`preprocessing.py:14-17`).

### 2.2 FID, precision, recall à 60k (runs sains seulement)

Toutes les moyennes de `summary.csv` sont retrouvées exactement depuis `runs.csv` (`stats.py`). **Seule différence : `summary.csv` donne l'écart-type de population** (`np.std`, ddof = 0, `make_data_figures.py:89`). L'écart-type d'échantillon (ddof = 1) est plus grand : 11,2 au lieu de 10,0 pour `data_ref`.

| Variante | Effondrés | FID 60k (sains) moy ± é.-t. (ddof=1) | P / R 60k | Welch vs ref (sains) | Médiane FID **tous** runs | Plafond vraies images FID / P / R |
|---|---|---|---|---|---|---|
| `data_ref` | 3/8 | 75,7 ± 11,2 | 0,753 / 0,718 | — | 87,0 | 2,9 / 0,864 / 0,892 |
| `sel_95` | 2/5 | 76,0 ± 10,8 | 0,775 / 0,680 | p = 0,97 | 88,4 | 3,1 / 0,882 / 0,863 |
| `sel_90` | 2/5 | 111,9 ± 19,6 | 0,793 / 0,646 | p = 0,067 | 127,3 | 5,0 / 0,893 / 0,842 |
| `sel_75` | 0/3 | 215,9 ± 48,2 | 0,837 / 0,563 | p = 0,034 | 219,8 | 9,8 / 0,917 / 0,765 |
| `sel_50` | 4/8 | 331,4 ± 15,4 | 0,843 / 0,382 | p < 0,001 | 1 333 | 25,3 / 0,942 / 0,630 |
| `sel_50_global` | 0/3 | 439,8 ± 15,5 | 0,801 / 0,356 | p < 0,001 | 431,0 | 180,0 / 0,923 / 0,654 |
| `sel_50_pixels` | 2/3 | 509,5 (1 run) | 0,882 / 0,331 | — | 2 095 | 21,8 / 0,933 / 0,640 |
| `noise_01` (0,1 → 0) | 4/8 | 64,6 ± 5,3 | 0,764 / 0,718 | p = 0,099 | 1 087 | = train |
| `noise_03` (0,3 → 0) | 0/3 | 67,7 ± 5,8 | 0,763 / 0,719 | p = 0,23 | 66,9 | = train |
| `noise_01_const` | 2/3 | 72,7 (1 run) | 0,799 / 0,700 | — | 2 244 | = train |
| `sel_50_noise_01` | 0/3 | 336,4 ± 24,6 | 0,859 / 0,392 | vs `sel_50` : p = 0,78 | 326,7 | comme `sel_50` |

Sources : `runs.csv:2-53`, `summary.csv:2-12`, `real_subsets.csv:2-8`, `stats_out.txt`.

**Autres lectures :**
- **FID final (120k)** : il remonte pour toutes les variantes, à 223–231 pour la référence et les variantes bruitées. Le bruit ne change rien à la fin : Welch p = 0,72 (σ 0,1) et p = 0,86 (σ 0,3), `stats_out.txt`.
- **Meilleur point de chaque run** (choisi sur le test, donc optimiste) : la sélection culmine plus tôt (36k–48k), mais reste bien pire. Exemple : `sel_50` 268 contre 72 pour la référence (`runs.csv`, colonne `fid_best`).
- **FID par chiffre** (`per_class_fid.csv`, G final) : `noise_03` 55,5 contre 71,7 (Welch p = 0,044 avec l'écart-type corrigé en ddof = 1 ; p = 0,024 brut). `sel_95` p = 0,078. Avec 8 comparaisons, le seuil de Bonferroni est 0,0063 : **aucune ne passe**.

### 2.3 Effondrements (vérifié)

- **15/34** (slide 9) : exact, mais seulement sur les 5 variantes de la figure (`data_ref`, `noise_01`, `sel_95`, `sel_90`, `sel_50`).
- Sur **toutes** les variantes sans σ = 0,3 : **19/49** (IC 95 % [0,25 ; 0,54]). Au total : 19/52, comme dans le rapport (ligne 15).
- **« 0 of 8 » avec bruit 0,3 n'est pas vérifié.** Seuls **3 runs** `noise_03` existent (`runs.csv:45-47`, `results_juliet.csv`). Les 5 autres sont des seeds `noise_03_e100` encore sur Juliet au moment du tracé : `make_slide_figures.py:5` (« EXPECTED : seeds encore sur Juliet, **comptés non effondrés** »), `:30` (`EXPECTED = {'Noise 0.3': 8}`), `:89` (`n = max(len(g), EXPECTED…)`). Aucune trace de ces runs n'existe sur les branches distantes (`git grep noise_03_e100` : seulement des configs et des scripts).

**Tests sur les effondrements (`stats.py`, `seed_table.py`) :**

| Comparaison | p (Fisher bilatéral) |
|---|---|
| 15/34 vs 0/8 (ce que dit la slide) | 0,035 |
| 15/34 vs **0/3** (ce qui est observé) | **0,26** |
| 19/49 vs 0/3 | 0,29 |
| Seeds 0–2 seulement : 10/30 vs 0/3 | 0,54 |
| `data_ref` 3/8 vs `noise_03` 0/3 | 0,49 |
| σ = 0,1 (toutes variantes) 6/14 vs `data_ref` 3/8 | 1,0 |
| Avec sélection 10/30 vs sans 9/22 | 0,77 |

- **Point décisif.** `data_ref` **n'a eu aucun effondrement sur ses seeds 0, 1 et 2** : ses effondrements sont aux seeds 4, 5 et 6 (`seed_table_out.txt`). Le 0/3 de `noise_03`, sur les seeds 0–2, est donc **exactement** ce qu'a fait la référence sur les mêmes seeds.
- Parmi les 5 autres variantes à 3 seeds sans σ = 0,3, trois font aussi 0/3 : `sel_75`, `sel_50_global` et `sel_50_noise_01`.
- Avec un taux de base de 39 %, 0/3 arrive par hasard avec une probabilité de 0,23.
- Pour qu'un « 0 sur n » devienne significatif contre 19/49, il faut **n ≥ 8** (p = 0,042) ; avec n = 10, p = 0,022 (`seed_table_out.txt`).

### 2.4 Mécanique de l'effondrement (vérifiée)

- **Sur le G final** (`saturation.csv`) :
  - Runs sains (33) : D(G(z)) ∈ [0,18 ; 0,38], gradient reçu par G ∈ [3,1 ; 11].
  - Runs effondrés (19) : gradient < 1,8·10⁻¹² pour les 19. D(G(z)) < 10⁻¹² pour 17 runs ; D(G(z)) = 1 pour 2 runs (`data_ref_s5`, `noise_01_const_s1`). Pré-activation de la tanh > 10 pour 16 runs (les exceptions sont `data_ref_s5`, `sel_95_s0` et `noise_01_const_s1`). Diversité selon z (`std_over_z`) ≤ 0,17.
  - Les chiffres du rapport (§8) sont donc exacts. `runs.csv` et `saturation.csv` sont cohérents run par run.
- **Cause numérique** (`noise_mechanics.py`, partie 1). Gradient de la perte de G par rapport au logit de D, avec sigmoïde puis `BCELoss`, comme dans le code fourni :

  | Logit | −27 | −28 | −30 | −40 | −80 | −100 |
  |---|---|---|---|---|---|---|
  | Gradient | 1,0 | 0,69 | 0,094 | 4,2·10⁻⁶ | 1,8·10⁻²³ | 0 |

  Avec `BCEWithLogitsLoss`, il vaut 1,0 partout. La cause est la borne ε = 10⁻¹² du *backward* de `BCELoss`, appliquée quand la sortie de D passe sous 10⁻¹², c'est-à-dire un logit < −27,6. **Vérifié.**
- **Non vérifié** : « D gagne dès l'époque 2 à 7 » (rapport §8) et « D wins less early » avec σ = 0,3 (slide 9). Les courbes D(x) par époque sont dans les fichiers TensorBoard sur Juliet, qui ne sont pas versionnés. Aucune courbe pour `noise_03` n'a été montrée.

### 2.5 Protocole commun (RpGAN + R1 + EMA, split 50k, 200 époques, 3 seeds, contre test)

| | Inception FID | CNN FID | CNN P / R | Effondrés |
|---|---|---|---|---|
| RpGAN + R1 + EMA | 22,14 ± 1,26 (23,56 ; 21,15 ; 21,69) | 9,93 ± 2,47 | 0,738 / 0,828 | 0/3 |
| + bruit 0,3 | 21,44 ± 0,58 (22,12 ; 21,07 ; 21,14) | 9,01 ± 1,04 | 0,738 / 0,825 | 0/3 |
| Welch | **p = 0,46** | **p = 0,60** | p = 0,88 / 0,79 | — |

Source : `cmp.csv`, lignes 8, 10, 12 et 14, 16, 18.

- La baseline commune (BCE) fait **0/3 effondrement** à 50k images : Inception 39,9 / 37,4 / 38,7 ; CNN 162 / 201 / 182 (`cmp.csv:2,4,6`).
- **Le bruit n'a jamais été mesuré en Inception FID sur la baseline BCE.** `configs/compare/noise_01.yaml` et `sel_50.yaml` existent dans le worktree, mais aucun résultat `cmp_*` n'existe.

### 2.6 Problème de reproductibilité trouvé (vérifié)

- Sur `quentin/comparaison` (le worktree), `utils.D_logits` lit `D.fc1` directement après `unwrap(D)`. Avec `InstanceNoise`, cela lève une erreur : `AttributeError: 'InstanceNoise' object has no attribute 'fc1'`, reproduite localement avec le code du worktree.
- **RpGAN + bruit plante donc sur cette branche.** La version qui marche est sur `noe/training` (`utils.py:25-35`, méthode `InstanceNoise.perturb`). C'est elle qui a produit `cmp.csv`.
- `main` (fbdd93f) ne contient ni `preprocessing.py` ni `D_logits`.
- **Toute relance doit partir de `noe/training`.**

---

## 3. Divergences et pourquoi

### 3.1 Sélection : même sens que le papier pour P et R, sens opposé pour le FID

| | DeVries (64×64, 50 %) | Nous (50 %, par chiffre) |
|---|---|---|
| ΔFID | −41 % (21,4 → 12,6) | **+338 %** (75,7 → 331,4) |
| ΔP | +0,11 | +0,09 |
| ΔR | −0,03 | **−0,34** |

1. **MNIST est déjà propre ; la queue est de la diversité.**
   - Sur les vraies images, garder 50 % fait passer le recall de 0,892 à 0,630 et le FID de 2,9 à 25,3 (`real_subsets.csv:2,6`). Avant tout entraînement, on s'impose donc un plafond de FID 9 fois plus haut.
   - Chez DeVries, le recall d'un GAN entraîné à 50 % ne perd que 0,03 à 0,04 (Tables 1 et 6), car la queue d'ImageNet est faite d'images atypiques que le modèle ne savait de toute façon pas produire.
   - (Vérifié pour les chiffres, interprétation = hypothèse appuyée par la figure *kept vs removed*.)
2. **La capacité n'est pas le facteur limitant chez nous.**
   - Le même G imposé atteint CNN FID 9,9 avec RpGAN + R1 + EMA (`cmp.csv`), contre 182 avec la BCE. Le goulot est la perte et la dynamique d'entraînement, pas la capacité.
   - L'argument de DeVries, « réallouer la capacité vers les zones denses », n'a donc rien à réallouer. Au contraire, le plafond de `sel_50` (25,3) est **pire** que ce que G atteint sur toutes les données avec une bonne perte (9,9).
   - (Vérifié sur les chiffres ; les deux plafonds viennent de protocoles proches mais non identiques : 10k images de train contre test.)
3. **G n'est pas conditionnel.**
   - Avec `sel_50`, le 0 tombe à 0,4 % des images générées et le 2 à 2,4 % (`summary.csv:6`, `classes_60k`), alors que les données vues restent équilibrées (`real_subsets.csv:6`).
   - À chiffre fixé, l'excès au-dessus du plancher est le même : 63 pour `sel_50` comme pour `data_ref` (rapport §6). La perte de FID global vient surtout du mélange des chiffres.
4. **Les modes deviennent plus isolés** (hypothèse du rapport). Le rapport distance au plus proche voisin d'un autre chiffre / distance au plus proche voisin du même chiffre passe de 3,60 à 5,16 dans les features (`class_gaps.csv:2,6`). Corrélation seulement.
5. **D sur-apprend plus.** `sel_50` voit chaque image 400 fois ; le rapport §9 montre que D mémorise ses images. C'est le même risque que DeVries signale dans son Annexe F. Comme le **contrôle « 50 % au hasard » n'a pas été fait**, on ne peut pas séparer l'effet « moins d'images » de l'effet « images choisies ».

**Circularité (vérifiée dans le code) :**
- La sélection utilise **exactement** les features du CNN FID : même fichier `mnist_classifier.pth`, même couche `fc` avant ReLU, mêmes fonctions `load_classifier` / `predict` (`preprocessing.py:14-17`, `evaluate.py:predict`, `evaluate.evaluate`). Le classifieur a en plus été entraîné sur ces mêmes 60k images.
- Conséquence : la sélection garde les points proches du centre de chaque classe **dans l'espace même où l'on mesure**. La precision k-NN monte, et le recall et la covariance baissent, de façon presque mécanique.
- Ici, le biais **amplifie l'ampleur du compromis mesuré, sans en inverser le sens**. Sur les vraies images, la sélection en pixels (ACP) donne P 0,933 / R 0,640 / FID 21,8, contre 0,942 / 0,630 / 25,3 en features CNN (`real_subsets.csv:6,8`). L'écart dû à la circularité est donc faible sur les données réelles (vérifié).
- Pour les GAN, `sel_50_pixels` n'a qu'un seul run sain : on ne peut rien conclure.
- La correction propre consiste à **sélectionner dans un espace et évaluer dans un autre** (§6).
- Phrasé à éviter : « la sélection ne dépend pas beaucoup du classifieur » (rapport §4). Cela est vrai sur les sous-ensembles réels, mais cela ne lève pas le biais de l'évaluation, comme le rapport le dit lui-même dans ses Limites.

### 3.2 Bruit : pas d'effet mesurable sur le FID, et la stabilité n'est pas démontrée

1. **Pourquoi le FID ne bouge pas.**
   - À 60k, σ vaut déjà 0 : le bruit a disparu au moment de la lecture.
   - À la fin, les distributions réelle et générée d'un run sain se recouvrent déjà. Le long de la direction de la différence des moyennes, la séparation vaut 0,53 écart-type pour `data_ref_s0` et 0,50 pour `noise_03_s0` ; un seuil linéaire n'y obtient qu'environ 60 % de précision (`noise_mechanics_out.txt`).
   - Le bruit n'a donc plus rien à « faire recouvrir » : c'est conforme à Arjovsky & Bottou (théorème 3.3).
2. **Pourquoi σ0 = 0,1, et même 0,3, ne suffit pas à rendre les supports « recouvrants » au début** (`noise_mechanics.py`, partie 2) :
   - Au départ (G à l'initialisation), les vraies et les fausses images sont séparées de |Δμ| = 22,6 (pixels dans [−1, 1]), avec un écart-type projeté de 1,05 : soit **21,5 écarts-types**.
   - Un bruit isotrope n'ajoute que σ dans chaque direction : avec σ = 0,1, la séparation reste à 21,4 ; avec σ = 0,3, à 20,7 ; même avec σ = 3, elle est encore de 7,1.
   - Le bruit est grand en norme (σ√784 ≈ 28σ), mais réparti sur 784 directions. **D n'a besoin que d'une direction**, où le bruit ne pèse presque rien. C'est exactement la remarque de Roth et al. sur le bruit en grande dimension.
   - Test direct : un D identique à `model.py`, entraîné 300 pas contre G à l'initialisation, atteint **100 % de précision quel que soit σ ∈ {0 ; 0,1 ; 0,3 ; 1}**. Le logit moyen sur les fausses images varie à peine : −10,1 (σ = 0) contre −10,7 (σ = 0,3) ; D(G(z)) vaut 5·10⁻⁵ contre 1,3·10⁻⁴.
   - **Conclusion (vérifiée sur ce test simplifié, avec G figé) :** l'explication « le bruit 0,3 empêche D de gagner » n'est pas soutenue par la géométrie. Pour vraiment faire recouvrir les distributions, il faudrait σ de l'ordre de 10, ce qui noierait le signal. **Hypothèse restante** : un petit effet de lissage de D (Roth : bruit ≈ pénalité de gradient) pourrait suffire à rester sous le seuil numérique de −28 dans la dynamique réelle. Ce n'est pas démontré.
3. **Sur RpGAN + R1, le bruit est redondant.** R1 est la version analytique du bruit d'instance (Roth 2017 ; Mescheder 2018). Le gain observé, 22,1 → 21,4 en Inception, n'est pas significatif (p = 0,46) et n'a que 3 seeds.

---

## 4. Ce qui marche / ne marche pas, et dans quel cas

| Méthode | Verdict chez nous | Dans quel cas elle pourrait marcher |
|---|---|---|
| Sélection 50–75 % | **Nuit nettement** : FID ×3 à ×4, recall −0,15 à −0,34, chiffres perdus (vérifié, p ≤ 0,034) | Données sales et très diverses, G conditionnel, capacité limitée (DeVries) |
| Sélection 90 % | Déjà moins bonne (FID 112 contre 76, p = 0,067, 3 runs sains) | — |
| Sélection 95 % | **Neutre** (76,0 contre 75,7, p = 0,97) | Utile seulement s'il y a de vraies erreurs d'étiquetage ou d'image à retirer |
| Sélection globale (sans séparer les chiffres) | **Pire** : le déséquilibre des classes donne un plafond FID de 180 | Jamais pour un G non conditionnel |
| Sélection en pixels | Même sous-ensemble ou presque (plafond 21,8 contre 25,3) ; 1 seul run GAN sain | Sert surtout à casser la circularité |
| Bruit 0,1 décroissant | Aucun effet démontré : FID p = 0,10 à 60k, p = 0,72 à la fin ; effondrements 4/8 | — |
| Bruit 0,1 constant | 2/3 effondrés, 1 run sain : rien à conclure | — |
| Bruit 0,3 décroissant | FID comme la référence (p = 0,23) ; **0/3 effondrement, non significatif** (p = 0,26 à 0,49) | Peut-être avec un σ plus grand ou adaptatif (Sønderby) ; à tester avec ≥ 8 seeds |
| Bruit 0,3 sur RpGAN + R1 + EMA | Inception 21,4 contre 22,1 (p = 0,46) | Redondant avec R1 ; coût nul, donc acceptable de le garder, mais **sans le présenter comme un gain** |
| Sélection + bruit 0,1 | Comme la sélection seule (p = 0,78) | — |

**Le vrai levier contre les effondrements** (vérifié) est la **perte** : soit `BCEWithLogitsLoss` (le gradient reste à 1), soit RpGAN + R1 (0/3 effondrement, CNN FID 9,9). Ce n'est pas le bruit.

---

## 5. Corrections des slides (`slides/comparaison.tex`, numéros = numéros affichés « n / 21 »)

| Slide (ligne .tex) | Actuel | Problème | Proposition |
|---|---|---|---|
| 1 « Three levers » (l. 47) | Instance noise — Sønderby et al. 2017 | Attribution correcte ; la justification théorique vient d'Arjovsky & Bottou | `Sønderby et al. 2017; Arjovsky \& Bottou 2017` (facultatif) |
| 2 « Protocol » (l. 61) | `120k updates, 3 to 8 seeds` | La lecture se fait à 60k ; après exclusion, il reste 3 à 5 runs par barre | `120k updates (read at 60k), 3 to 8 seeds` |
| 6 « Data: selection » (l. 91-92) | `f(x): CNN features of x` | Il manque la circularité | `f(x): features of the same CNN as the CNN FID` (+ `\source{…selection and metric share the CNN: biased toward precision}`) |
| 6 (l. 95) | `Removed images are rare styles, not errors.` | Jugement fait sur 10 images par chiffre | `…mostly rare styles (10 lowest per digit shown)` |
| 7 « Data: noise » (l. 101-102) | `falls linearly from σ0 to 0 over 100 epochs` | Exact, mais c'est un escalier par époque, et σ = 0 au point de lecture | Ajouter `(= 60k updates: no noise left at the 60k reading point)` |
| 8 « Data: results » (l. 111) | `Noise: same as the reference` | Juste, mais les barres montrent 65 et 68 contre 76 | `Noise: within seed spread (Welch p = 0.10, 0.23)` |
| 8 (l. 112) | `3 to 8 seeds, collapsed runs excluded` | Biais du survivant non chiffré | `3 to 5 healthy runs per bar (collapsed runs excluded, see next slide)` |
| 9 « Data: collapsed runs » (l. 126) | `15 of 34 runs collapse without noise 0.3` | Sous-ensemble choisi ; le total est 19/49 | `19 of 49 runs collapse without noise 0.3` (ou garder 15/34 en précisant « in the 5 configs shown ») |
| 9 (l. 126) | `D wins, its sigmoid saturates, G gets no gradient` | La cause est numérique (BCELoss sur sigmoïde), pas la théorie | `D wins; with sigmoid + BCELoss, G's gradient underflows (logit < −28)` |
| 9 (l. 128) | `Noise 0.3: no collapse (0 of 8): D wins less early` | **Faux en l'état** : 3 runs observés, 5 comptés non effondrés par défaut (`make_slide_figures.py:30,89`). « D wins less early » n'est pas mesuré, et `data_ref` faisait aussi 0/3 sur les seeds 0–2 | `Noise 0.3: 0 of 3 (reference: 0 of 3 on the same seeds; not significant)`. Si les 5 seeds rapatriées sont saines : `0 of 8 (Fisher p = 0.04 vs 19/49)` |
| 9 (figure `data_collapse.png`) | Barre « 0 / 8 » | Même problème | Retracer après avoir vidé `EXPECTED`, ou afficher « 0 / 3 » |
| 21 « Conclusion » (l. 228) | `+ noise 0.3 & 21.4 & 9.0` | Présenté comme le meilleur chiffre, mais non significatif | Ajouter `\source{… noise: p = 0.46 (Inception), 0.60 (CNN), 3 seeds}` |
| 21 (l. 231) | `Noise 0.3: fewer collapses.` | Non démontré (0/3 contre 3/8, p = 0,49 ; avec RpGAN, 0/3 dans les deux bras) | `Noise 0.3: no measurable effect.` |

**Notes orales (`oral-data.md`) :**
- l. 16 : préciser « de 17,2 à 9,1 en 64×64, à 40 % de rétention (Table 6) ».
- l. 22 : `15 runs sur 34` → `19 sur 49`.
- l. 26 : `aucun run ne s'effondre (0 sur 8) : le bruit empêche D de gagner trop vite` → `0 sur 3, comme la référence sur les mêmes seeds : non concluant`.
- Dernière puce : « chez nous, le bruit stabilise l'entraînement » → « nous n'avons pas pu montrer que le bruit stabilise ».

**Rapport de Nicodème** (à garder cohérent avec les slides) :
- Les écarts-types de la table §5 sont en ddof = 0 : passer en ddof = 1 (par ex. 75,7 ± 11,2).
- Le rapport lui-même dit que « le 0 sur 3 de `noise_03` … ne prouve rien » (l. 244). **Les slides le contredisent.**
- Ajouter le taux d'effondrement global (19/49 sans σ = 0,3) dans la conclusion de chaque variante.

---

## 6. Expériences à relancer (Juliet)

Toutes les commandes se lancent depuis la racine du dépôt sur Juliet, avec `export DATA=/projects/m25146/data/; source venv/bin/activate`. La branche est indiquée pour chaque ligne. Coût estimé à partir des horodatages de `results_juliet.csv` : environ 30–35 min GPU par run Données de 120k mises à jour (évaluation CNN seulement), environ 15 min pour 100 époques. Les runs du protocole commun avec Inception et VGG sont prévus pour moins de 2 h (`compare.sh`).

**Modèle sbatch (copié de `scripts/data_runs.sh`) :**
```bash
J='sbatch -p mesonet -N 1 -c 8 --gres=gpu:1 --mem=32G --account=m25146 --output=outputs/%x_%A_%a.out'
```

| Prio | Expérience | Pourquoi | Coût | Commande |
|---|---|---|---|---|
| **P0** | Rapatrier ou analyser les 5 seeds `noise_03_e100` déjà lancées, puis vider `EXPECTED` | La slide 9 en dépend | 0 GPU | `grep noise_03_e100 results.csv` ; `ls runs/noise_03_e100_s*` ; puis `python figures/make_slide_figures.py` avec `EXPECTED = {}` (`nico/preprocessing`) |
| **P0** | Courbes D(x) des époques 1–10 depuis les fichiers TensorBoard existants (52 runs) | Vérifier « D gagne à l'époque 2–7 » et « D wins less early » avec σ = 0,3 | 0 GPU, CPU quelques minutes | Petit script `EventAccumulator(run).Scalars('D(x)')` sur `runs/*/events.out.tfevents.*` (même lecteur que `make_data_figures.py:75`) |
| **P1** | `noise_03_e100` seeds 3–12 et `data_ref` seeds 8–12 (100 époques suffisent : les effondrements arrivent tôt) | Avoir assez de seeds pour un test d'effondrement (n ≥ 8 nécessaire, §2.3), avec des seeds appariées | 15 runs × ~15 min ≈ 4 GPU·h ; ~1 h avec 4 en parallèle | `$J --time=02:00:00 --array=3-12%4 --wrap 'export DATA=/projects/m25146/data/; source venv/bin/activate; python train.py --config configs/data/noise_03_e100.yaml --seed $SLURM_ARRAY_TASK_ID'`, puis la même chose avec un `data_ref_e100.yaml` (copie de `data_ref.yaml` avec `epochs: 100`, `save_every: 100`) et `--array=8-12` (`nico/preprocessing`) |
| **P1** | **BCE + bruit 0,3 au protocole commun** (split 50k, 200 époques, Inception + VGG), seeds 0–2 (idéalement 0–7) | La case manquante : le bruit n'a jamais été mesuré en Inception sur la baseline BCE | 3–8 runs × < 2 h | Sur `noe/training` : `sed 's/cmp_noise_01/cmp_noise_03/; s/sigma: 0.1/sigma: 0.3/' configs/compare/noise_01.yaml > configs/compare/noise_03.yaml` puis `$J --time=02:00:00 --array=0-7%4 --wrap 'export DATA=/projects/m25146/data/; source venv/bin/activate; python train.py --config configs/compare/noise_03.yaml --seed $SLURM_ARRAY_TASK_ID'` ; la baseline `configs/compare/baseline.yaml` avec les mêmes seeds pour l'appariement |
| **P2** | Casser la circularité, sans entraînement : FID Inception et P/R VGG des **sous-ensembles réels** (`all`, `sel_95/90/50`, `sel_50_pixels`) | Vérifier si « precision ↑, recall ↓ » tient dans un espace indépendant du CNN de sélection | < 20 min GPU | Sur `noe/training` : `scripts/run.sh "python -c \"from preprocessing import *; from evaluate import *; from data import mnist_split; import torch; x,y=mnist_split('train'); f=embed(x,'classifier','cuda'); [print(k, evaluate_prof(x[select_instances(y,f,k)][torch.randperm(int(k*len(x)))[:10000]],'cuda','test')) for k in (1.0,0.95,0.9,0.5)]\"" 00:30:00` (à adapter si `evaluate_prof` attend un autre format) |
| **P2** | Casser la circularité, avec entraînement : `sel_90` et `sel_95` **sélectionnés en pixels**, évalués en Inception et en CNN | Sélection et métrique dans deux espaces différents | 6 runs × < 2 h | Sur `noe/training` : `configs/compare/sel_90_pixels.yaml` = `baseline.yaml` + `data: {selection: {enabled: true, keep: 0.9, per_class: true, space: pixels}}` et `epochs: 222` (même nombre de mises à jour) ; idem `keep: 0.95`, `epochs: 211` ; `$J --array=0-2 --wrap '… python train.py --config configs/compare/sel_90_pixels.yaml --seed $SLURM_ARRAY_TASK_ID'` |
| **P2** | Contrôle **50 % au hasard** (le contrôle de DeVries, Table 1) | Séparer « moins d'images » de « images choisies » | 3 runs × 30 min | Nécessite un patch de 2 lignes dans `preprocessing.embed` : `if space == 'random': return torch.randn(len(x), 1, generator=torch.Generator().manual_seed(0))`, puis `configs/data/sel_50_random.yaml` (`space: random`, `epochs: 400`) |
| **P3** | Sélection 90–95 % **sur RpGAN + R1 + EMA** | DeVries (Annexe F) suggère R1 comme complément ; les plafonds de `sel_95` (3,1) et `sel_90` (5,0) sont sous le 9,9 atteint, donc ne bloquent pas | 6 runs × < 2 h | Sur `noe/training` : `configs/compare/rpgan_r1_ema.yaml` + section `data.selection` (`keep: 0.95`, `space: pixels`) |
| **P3** | Plus de seeds pour `sel_95` et `sel_90` (seeds 5–7) | 3 runs sains seulement chacun | 6 × 30 min | `$J --time=08:00:00 --array=5-7 --wrap '… python train.py --config configs/data/sel_95.yaml --seed $SLURM_ARRAY_TASK_ID'` (idem `sel_90`) |
| **P3** | Bruit plus fort (σ0 = 1) ou adaptatif (Sønderby) sur BCE | Tester si un σ assez grand empêche la saturation (le §3.2 indique qu'il faudrait un σ beaucoup plus grand) | 5 × 15 min | `configs/data/noise_10_e100.yaml` (copie de `noise_03_e100` avec `sigma: 1.0`) |

**Ce qu'il ne faut pas relancer :** `sel_50`, `sel_50_global` et `sel_75`. Le résultat est net : FID ≥ 216, Welch p ≤ 0,034.

---

## 7. Approfondissements possibles

1. **Intégrer les effondrements dans la métrique.**
   - Rapporter le couple (taux de succès, FID des runs sains), ou la **médiane sur tous les runs** (§2.2), qui ne cache rien.
   - Pour la décision, « P(un run est utilisable) » vaut 61 % (IC 95 % [46 % ; 75 %]) avec la BCE, contre 100 % (3/3) avec RpGAN + R1.
2. **Analyse appariée par seed.** À seed égale, G et D partent des **mêmes poids** dans toutes les variantes : `set_seed` est appelé avant la construction des modèles, et la sélection ne consomme pas le générateur aléatoire global (vérifié dans `train.py`). Un test de McNemar sur des seeds appariées serait plus puissant que Fisher.
   - Attention : le bruit, et les z tirés sur le GPU, décalent ensuite le flux de tirages aléatoires.
   - Dans le `train.py` de Nicodème, `generate()` pendant l'évaluation consomme aussi le générateur aléatoire global, donc **évaluer modifie l'entraînement**. Ce point est corrigé par `fork_rng` dans le code commun.
3. **Le bruit comme pénalité de gradient.** Mesurer ‖∇ₓD‖ sur les vraies images au cours de l'entraînement, avec et sans σ, pour tester l'hypothèse de Roth (bruit ≈ R1 de force ∝ σ²).
4. **Rééquilibrer après la sélection** (piste du rapport) : par exemple un tirage par chiffre prédit (rejet par classe) au moment de la génération, pour isoler l'effet « qualité à chiffre fixé ».
5. **Densité / couverture** (Naeem 2020), comme dans DeVries : des mesures moins sensibles aux points isolés que P/R k-NN, utiles pour le compromis qualité / diversité.
6. **Point de lecture.** Choisir 60k mises à jour sur le **test** à partir de `data_ref` biaise en faveur de la référence : la sélection culmine plus tôt. Refaire ce choix sur la **validation** du split commun, ou comparer au meilleur checkpoint choisi sur la validation.

---

## 8. Sources

**Code et données (lecture seule) :**
- `origin/nico/preprocessing` (56531a4) : `preprocessing.py`, `train.py`, `evaluate.py`, `utils.py`, `model.py`, `configs/data/*.yaml`, `scripts/data_runs.sh`, `figures/make_slide_figures.py` (l. 5, 30, 89), `figures/make_data_figures.py` (l. 77, 83, 89), `figures/data/*.csv`, `rapport-donnée-illustré.md` (l. 15, 125-135, 140-144, 196-197, 244).
- `wt-comparaison` (= `origin/quentin/comparaison`, 7474210) : `results/comparison_local.csv`, `configs/compare*/`, `utils.py:25-30` (`D_logits` sans gestion du bruit), `preprocessing.py:74-82`.
- `origin/noe/training` (44ad088) : `utils.py:25-35` (`D_logits` avec `perturb`), `preprocessing.py` (`InstanceNoise.perturb`).
- `main` (fbdd93f) : `slides/comparaison.tex`, `slides/oral-data.md`, `slides/figures/data_*.png`.

**Scripts de vérification** (`C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad\scripts\data\`) :
- `stats.py` → `stats_out.txt` : moyennes, Welch, Mann-Whitney, Fisher, saturation, protocole commun.
- `seed_table.py` → `seed_table_out.txt` : table seed × variante, puissance du test de Fisher.
- `noise_mechanics.py` → `noise_mechanics_out.txt` : gradient BCE, séparation linéaire, course de D.

**Papiers :**
- T. DeVries, M. Drozdzal, G. W. Taylor, *Instance Selection for GANs*, NeurIPS 2020 (arXiv 2007.15255). PDF local `papiers/preprocessing/`. Tables 1, 2, 3, 6 ; §4.3, §4.4 ; Annexe F.
- C. K. Sønderby, J. Caballero, L. Theis, W. Shi, F. Huszár, *Amortised MAP Inference for Image Super-resolution*, ICLR 2017 (arXiv 1610.04490v3). §3.2.1, Annexes C et D.
- M. Arjovsky, L. Bottou, *Towards Principled Methods for Training GANs*, ICLR 2017 (arXiv 1701.04862). Th. 2.1, 2.2, 2.4, 2.6, 3.1–3.3.
- K. Roth, A. Lucchi, S. Nowozin, T. Hofmann, *Stabilizing Training of GANs through Regularization*, NeurIPS 2017 (arXiv 1705.09367). §2, §3, §5.4.
- L. Mescheder, A. Geiger, S. Nowozin, *Which Training Methods for GANs do actually Converge?*, ICML 2018 (arXiv 1801.04406). Résumé.
