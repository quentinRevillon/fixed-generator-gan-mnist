# FID / évaluation : pourquoi un « FID CNN » à côté du FID Inception, et ce que chacun mesure vraiment

> **En une phrase :** les deux FID ne mesurent pas la même chose. Le FID CNN mesure surtout **le mélange des chiffres** : 67 % de la variance de ses features est due au chiffre (26 % pour Inception), et les seules proportions de la baseline expliquent 77 % de son FID CNN contre 13 % de son FID Inception. Le FID Inception mesure surtout **l'aspect des images** : du bruit sur 25 % des images pèse autant que tout le défaut de la baseline, alors qu'il est presque aveugle à un chiffre manquant (0,9 %). D'où le ÷18 contre ÷1,75 : RpGAN corrige d'abord le mélange des chiffres. **Recommandation :** FID Inception en mesure principale (c'est celle de la plateforme, la plus stable d'une seed à l'autre, comparable aux papiers). Le FID CNN sert de mesure de diagnostic des chiffres, avec l'histogramme des chiffres et un FID par chiffre.
>
> Livrables : scripts dans `scratchpad\scripts\fid\`, chiffres dans `scratchpad\results\fid\` (chemins complets en §8). Coût réel : environ 40 min de CPU (Inception à environ 4 images/s avec 2 threads, d'où N = 500 pour Inception).

**Lexique (à lire d'abord).**
- **Features (représentation)** : le vecteur de nombres qu'un réseau pré-entraîné associe à une image. Le FID ne compare jamais des pixels, il compare ces vecteurs.
- **FID** (Fréchet Inception Distance, Heusel 2017) : on résume les features des vraies images et celles des images générées par leur moyenne μ et leur matrice de covariance Σ (une « gaussienne »), puis `FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})`. Premier terme : les centres des deux nuages ; second : leurs formes. Plus bas = plus proche.
- **FID Inception** : features = dernière couche de pooling d'Inception-v3 entraîné sur ImageNet (photos couleur, 1 000 classes d'objets), 2 048 nombres. C'est le FID standard **et celui de la plateforme du cours**.
- **FID CNN** : même formule, features = couche de 128 nombres d'un petit CNN que le groupe a entraîné à reconnaître les chiffres MNIST (98,7 % de précision sur test), prise **avant** la dernière ReLU (`evaluate.py:61-62`).
- **Plancher** : FID entre deux échantillons de **vraies** images. C'est le meilleur score atteignable ; il n'est pas 0 car les échantillons sont finis.
- **Biais lié à N** : avec N images, le FID estimé est systématiquement plus grand que le vrai FID ; l'écart diminue quand N augmente (Chong & Forsyth 2020).
- **Perte de modes** : le GAN ne produit plus certains chiffres (un « mode » = un chiffre ici). **Déséquilibre** : il les produit en mauvaises proportions.
- **Spearman (ρ)** : corrélation des **rangs** ; ρ = 1 si les deux mesures classent les modèles dans le même ordre.

Convention : **[Vérifié]** = mesuré ou lu dans un fichier cité ; **[Hypothèse]** = interprétation non testée.

---

## 1. Attendu (littérature et cours)

**Le cours.**
- L'énoncé définit le FID avec Inception-v3, couche de pooling, dimension 2048 (`enonce/diapos_enonce.pdf`, slide 8). **[Vérifié]**
- Slide 9, « FID falls short » : *« Feature-dependent: Inception features may miss details that matter to us »*, et *« One number mixes quality and diversity »*. Le cours lui-même dit donc que le choix de l'extracteur compte. **[Vérifié]**
- Slide 33, « On the platform » : la plateforme note avec *« FID, Precision (for finite support), Recall (for finite support), (Obviously) the visual inspection »*. Slide 83 : *« Manifold uniformity is not the same as semantic class balance »*. Slide 86 : *« Explain what the measurements mean »*. **[Vérifié]**
- La page de résultats de la plateforme précise : *« Historical FID uses Inception features on all images; historical P/R uses VGG features with k=5 and 9,984 images »* (citée dans `architecture-choix.md:148`). **La métrique de la plateforme est donc le FID Inception, pas le FID CNN.** Notre `--metric prof` (`evaluate_prof`, `evaluate.py:254-262`) l'approche : `checkpoints/G.pth` obtient 44,0 au classement et 43,2 chez nous (contre `val`, `architecture-choix.md:150`, `baseline.md:178`). **[Vérifié]**

**Ce que dit la littérature sur l'extracteur.**
- **Le FID est un FID « Inception » par convention, pas par nécessité.** Heusel 2017 prend Inception parce qu'il évalue des photos ; la formule est générale. **[Vérifié, Heusel 2017]**
- **Inception est proche des classes ImageNet.** Kynkäänniemi et al. 2023 (ICLR) : l'espace du FID est *« so close to the ImageNet classifications »* qu'aligner les histogrammes des classes ImageNet prédites suffit à baisser fortement le FID **sans améliorer les images** ; *« FID is prone to intentional or accidental distortions »*. Conséquence pour nous : un FID réagit surtout aux **classes de son extracteur**. Celles d'Inception (objets ImageNet) ne contiennent aucun chiffre ; celles du petit CNN sont exactement les 10 chiffres. **[Vérifié pour le papier ; transposition à MNIST = Hypothèse, testée en §6]**
- **Inception est critiqué hors de son domaine.** Stein et al. 2023 (NeurIPS) : *« no existing metric strongly correlates with human evaluations »*, une cause étant l'*« over-reliance on Inception-V3 »* ; ils recommandent DINOv2. Jayasumana et al. 2024 (« Rethinking FID », arXiv 2401.09603) : représentation pauvre d'Inception, hypothèse gaussienne fausse, résultats *« inconsistent … when varying the sample size »* ; ils proposent CMMD (features CLIP + MMD). **[Vérifié]**
- **Remplacer Inception par un classifieur MNIST est une pratique publiée.** Bińkowski et al. 2018 (ICLR, « Demystifying MMD GANs », §4) : *« For models on MNIST, we replace the Inception featurization with features from a LeNet-like convolutional classifier »*. Hardy et al. 2019 (MD-GAN, arXiv 1811.03850) : *« we use a classifier more adapted to compute the FID on the MNIST dataset »*. Dedeoglu et al. 2021 (arXiv 2101.09225, annexe D) : *« 'ILSVRC 2012' dataset does not contain MNIST classes »*, ils entraînent un réseau MNIST (99,23 %) pour un « modified FID » qui sépare mieux leurs méthodes qu'Inception. Issenhuth et al. 2021 (déjà cité par le groupe, `architecture-choix.md:160-166`) utilise un classifieur MNIST pour précision/rappel. **[Vérifié]**
- **Mais Inception fonctionne déjà sur MNIST.** Lucic et al. 2018 : plancher Inception MNIST de 1,25 à N = 10 000 (fig. 1a), et *« FID is heavily influenced by the missing modes »* sur MNIST (fig. 1b). Heusel 2017 (fig. 3) : le FID Inception croît de façon monotone avec bruit, flou, rectangles noirs, etc. (sur CelebA). **[Vérifié]**
- **Le FID est biaisé et dépend de N.** Bińkowski 2018 : *« the estimator of FID is biased »* (le KID, lui, a un estimateur sans biais). Chong & Forsyth 2020 : le biais dépend du générateur, *« This effect cannot be fixed by evaluating at a fixed number of samples »*. Parmar et al. 2022 (clean-fid, arXiv 2104.11222) : le redimensionnement et la compression font varier fortement le FID — pertinent car Inception doit agrandir nos 28×28 en 299×299. **[Vérifié]**
- **Précision / rappel** (Kynkäänniemi 2019) séparent qualité et diversité ; Naeem et al. 2020 (ICML) les jugent *« not robust against outliers »* et proposent densité / couverture. **[Vérifié]**

**Attendu, en résumé.** Un FID calculé avec des features MNIST devrait être plus sensible aux défauts propres aux chiffres (chiffre manquant, chiffre ambigu) ; Inception devrait réagir davantage aux défauts « d'image » (bruit, flou, texture). Le prix : un FID CNN n'est comparable à aucun papier ni au classement.

---

## 2. Observé chez nous

### 2.1 Ce qui existe dans le dépôt (lecture seule)

- **Deux mesures, deux tableaux.** `cnn` (petit CNN, k = 3, `results/results.csv`) et `prof` (Inception + VGG k = 5, `results/results_prof.csv`) (`evaluate.py:3-5`, `:273`). **« prof » = notre reproduction de la métrique de la plateforme** (FID Inception sur toutes les images, P/R VGG k = 5 sur 9 984 images). Elle reste une approximation : le prétraitement et la référence de la plateforme ne sont pas publiés (`evaluate.py:201-202`). **[Vérifié]**
- **Le groupe avait déjà conclu :** « Choisir un modèle : FID Inception » et « Citer une valeur de FID seule : jamais avec le CNN » (`baseline.md:101-104`). Le CNN donne 142 à 224 pour les mêmes images selon la seed d'entraînement du CNN (`baseline.md:87-93`). **[Vérifié]**
- **Ce que les slides ont mesuré avec quel FID** (`slides/comparaison.tex:57-63`, dépôt principal) : Baseline et Sampling avec les deux FID contre val ; **Data et Training avec le CNN seul**, contre test ; protocole commun avec les deux FID, contre test. Le γ de R1 est choisi au FID CNN (`comparaison.tex`, slide « Training: choosing γ », source « small CNN FID »). **[Vérifié]**
- **Circularité, confirmée.** `preprocessing.py:19-22` et `:56` : la sélection d'instances de Nicodème calcule un score gaussien (distance de Mahalanobis par chiffre) **dans les features du même CNN** (`space: classifier` par défaut). Ses résultats sont ensuite mesurés avec le FID, la précision et le rappel **de ce même CNN**. Nicodème le signale (`rapport-donnée-illustré.md:313` sur `origin/nico/preprocessing`). Sa variante `sel_50_pixels` (ACP des pixels) donne un sous-ensemble de vraies images voisin (FID 21,8 contre 25,3, `rapport-donnée-illustré.md:107`), mais **elle ne lève pas le biais de l'évaluation**. `architecture-choix.md:171` (« aucun choix n'est fait avec lui ») est donc faux : la sélection et le choix de γ sont faits avec ce CNN. **[Vérifié]**

### 2.2 Les chiffres des slides, revus par seed

Source : `results/comparison_local.csv` (worktree), 3 seeds d'entraînement par méthode, 10 000 images contre test. **[Vérifié]**

| Méthode | FID Inception (moy ± é.-t. ; min-max) | FID CNN (moy ± é.-t. ; min-max) | Écart des chiffres* |
|---|---|---|---|
| Baseline | 38,7 ± 1,0 ; 37,4-39,9 | **181,8 ± 15,8 ; 162,3-201,0** | 25,9 % |
| Baseline + DRS | 34,4 ± 0,6 | 154,0 ± 25,2 ; 121,2-182,5 | 23,8 % |
| RpGAN + R1 + EMA | 22,1 ± 1,0 | 9,9 ± 2,0 ; 7,6-12,5 | 4,9 % |
| + DRS | 22,0 ± 0,9 | 14,2 ± 1,9 | 5,9 % |
| + bruit 0,3 | 21,4 ± 0,5 | 9,0 ± 0,8 | 4,8 % |

\*Écart : part des images qui devraient changer de chiffre pour revenir à 10 % par chiffre (définition de `baseline.md`).

- **Variation relative d'une seed à l'autre** (écart-type / moyenne) : Inception 2,6 % (baseline) et 4,5 % (RpGAN) ; CNN 8,7 % et 20 %. **Le FID CNN est 3 à 4 fois plus bruité.**
- **Corrélation de rang** (`scratchpad\results\fid\rank_correlation.csv`, script `rank_correlation.py`) :

| Ensemble apparié | n | ρ de Spearman | Lecture |
|---|---|---|---|
| Baseline, toutes les époques (`runs/baseline_200ep_prof/metrics.jsonl`) | 100 | +0,52 | d'accord seulement sur « début mauvais, fin meilleure » |
| Baseline, époques ≥ 100 | 51 | **−0,885** | **ordre inverse** : quand l'un baisse, l'autre monte |
| Protocole commun, 18 lignes | 18 | +0,81 | porté par l'écart baseline / RpGAN |
| Protocole commun, RpGAN seulement | 12 | +0,43 (p = 0,17) | **pas d'accord significatif entre bons modèles** |
| Balayages DRS sur la baseline (`results.csv` × `results_prof.csv`) | 18 | +0,02 | aucun accord |
| Écart des chiffres vs FID CNN | 18 | **+0,94** | le FID CNN suit le mélange des chiffres |

### 2.3 Expériences de validation exécutées (détail en §6)

Principe : on prend de **vraies** images MNIST, on leur applique un défaut connu et dosé, et on regarde de combien chaque FID bouge. Les mêmes images passent dans les deux réseaux. Pour comparer deux FID qui n'ont pas la même échelle, on exprime chaque défaut en **part de l'excès de FID de la baseline** : `(FID − plancher) / (FID baseline GAN − plancher)`, dans la même mesure. 100 % signifie que ce défaut pèse autant que tout ce qui sépare la baseline de vraies images, selon ce FID.

![Figure clé](../results/fid/fid_sensitivity.png) — `scratchpad\results\fid\fid_sensitivity.png` (script `make_figure.py`, données `audit.csv`).

**Résultat clé** (N = 500 images, mêmes images pour les deux réseaux ; colonne « CNN 10k » = même défaut au protocole du projet, N = 10 000). Source : `scratchpad\results\fid\audit.csv`, numéro de ligne entre crochets. **[Vérifié]**

| Défaut | Inception | CNN (N = 500) | CNN (N = 10 000) | Qui le voit le mieux |
|---|---|---|---|---|
| Bruit σ = 0,3 sur 25 % des images | **102 %** [75] | 13 % [45] | 9 % [11] | Inception (×8) |
| Flou σ = 1 sur 25 % | 18 % [79] | 29 % [49] | 23 % [15] | les deux, CNN un peu plus |
| Trait effacé (carré 10 px) sur 25 % | 18 % [87] | 16 % [57] | 12 % [23] | égalité |
| Chiffres « chimères » (moyenne de 2 chiffres différents) sur 25 % | 18 % [83] | 25 % [53] | 18 % [19] | égalité |
| 1 chiffre absent sur 10 | **0,9 %** [90] | 22 % [64] | 27 % [30] | CNN (×25) |
| 2 chiffres absents | 6 % [91] | 51 % [65] | 52 % [31] | CNN (×8) |
| 5 chiffres absents | 23 % [92] | 189 % [66] | 185 % [32] | CNN (×8) |
| 30 % de « 1 » (au lieu de 11 %) | 13 % [93] | 56 % [67] | 51 % [33] | CNN (×4) |
| **Proportions de la baseline** (25 % de « 1 », 1,6 % de « 2 ») | **13 %** [95] | 73 % [69] | **77 %** [35] | CNN (×6) |
| GAN baseline, **chiffres remis en proportion** | **116 %** (pire) [72] | 23 % [38] | 18 % [4] | — |
| GAN RpGAN + R1 + EMA | 56 % [73] | 1,4 % [39] | 6 % [5] | — |

Lecture en clair :
- **Pour 25 % d'images bruitées, le FID Inception augmente de 285 %** (14,65 → 56,3 ; [70], [75]). **Le FID CNN, lui, augmente de 108 %** (18,9 → 39,2 ; [36], [45]), une hausse noyée dans son propre bruit (écart-type du plancher 3,8, soit z = 5 contre z = 333 pour Inception).
- **Quand un chiffre sur dix disparaît, le FID Inception augmente de 2 %** (14,65 → 15,00 ; [90]), presque rien. **Le FID CNN augmente de 181 %** (18,9 → 52,9 ; [64]). À N = 10 000 : 2,7 → 45,2 [30], soit ×17.
- **Tous les défauts sont monotones pour les deux FID** : plus on dégrade d'images (10, 25, 50, 100 %), plus le FID monte (lignes [6]-[29], [40]-[63], [74]-[89]). Les deux sont des mesures valides ; elles n'ont pas la même sensibilité.

**Autres mesures exécutées** (toutes **[Vérifié]**) :
- **Part de la variance des features expliquée par le chiffre** (variance entre chiffres / variance totale, 2 000 images de test, `variance_share.csv`) : **CNN 0,669 ; Inception 0,257.**
- **FID CNN décomposé** (`cnn_decomposition.csv`, baseline seed 0, N = 10 000) : 162,3 = 68,0 (terme des moyennes) + 94,4 (terme des covariances). Une fois les chiffres remis en proportion : 31,6 = 8,7 + 22,9.
- **FID CNN « par chiffre »** (chaque chiffre généré comparé au même chiffre réel, 500 images par chiffre, moyenne sur les 9 chiffres autres que « 2 ») : vraies images 14,4 ; baseline 58,2 ; RpGAN 53,5. **À chiffre fixé, RpGAN ne gagne que 11 % de l'excès** ((58,2 − 14,4) → (53,5 − 14,4)). Le « 2 » : réel 29,0, baseline 199,6, RpGAN 114,8.
- **Planchers** (`audit.csv` [96]-[102]) : CNN à 10 000 images : train vs test 2,69, val vs test 2,93, train vs val 1,68. Le repo donne 3,1 et 1,7 (`architecture-choix.md:206-207`), même ordre de grandeur. Inception à 500 images : train vs test 14,39, val vs test 14,37.
- **Biais selon N** (même N pour la référence et l'ensemble mesuré, 5 tirages, [103]-[119]) :

| N | 250 | 500 | 1 000 | 2 000 | 5 000 | 10 000 |
|---|---|---|---|---|---|---|
| Inception, réel vs test | 29,7 ± 1,2 | 18,5 ± 0,2 | 11,3 ± 0,1 | — | — | ≈ 1,2 (vs val, `baseline.md:34`) |
| Inception, baseline GAN | 70,4 | 60,0 | — | — | — | 39,9 (seed 0, `comparison_local.csv`) |
| CNN, réel vs test | 62,7 ± 15,9 | 22,9 ± 4,1 | 17,1 ± 1,2 | 8,0 ± 0,8 | 4,1 ± 1,0 | 2,9 ± 0,3 |
| CNN, baseline GAN | 215 ± 56 | 198 ± 14 | 171 ± 9 | 175 ± 19 | 171 ± 9 | 162 ± 2 |

- **Avant / après ReLU** (`relu.csv`) : après la ReLU, 41 des 128 dimensions valent toujours 0 sur test et la covariance tombe au rang 87. Le FID se calcule quand même (plancher 1,56 ; baseline 83,6 ; RpGAN 9,2), dans le même ordre. Seule l'échelle change : baseline/RpGAN = 9,1 au lieu de 13,0.
- **Pixels gris** (`pixel_stats.csv`, part des pixels entre 0,05 et 0,5) : réel 4,96 %, baseline 5,94 %, RpGAN 5,66 %. L'écart est faible et ne tranche pas l'hypothèse d'un « fond sale » vu par Inception.
- **Contrôle du calcul :** `analyze.py` utilise une version rapide du FID, mathématiquement identique (valeurs propres). Elle redonne `evaluate.fid` à 10⁻⁴ près (`fast_fid_check.csv` : 2,6942/2,6942 ; 171,0918/171,0918 ; 55,5549/55,5548). Nos FID CNN des GAN redonnent ceux du dépôt : baseline s0 162,309 ([3]) et 162,309 (`comparison_local.csv`), RpGAN s0 12,517 et 12,517.

---

## 3. Divergences et pourquoi

**3.1 Pourquoi ÷18 en CNN et ÷1,75 en Inception ?**

1. **Les features du CNN sont d'abord des « étiquettes de chiffre ».** Le CNN a été entraîné à séparer les 10 chiffres, donc 67 % de la variance de ses features vient du chiffre (contre 26 % pour Inception). Un mélange de chiffres déséquilibré déplace la moyenne et déforme la covariance de ses features. Les deux termes du FID explosent. C'est la version MNIST de l'effet décrit par Kynkäänniemi 2023 pour les classes ImageNet. **[Vérifié par la mesure ; le lien avec le papier est une interprétation]**
2. **Le défaut principal de la baseline, pour le CNN, est son mélange de chiffres.** Avec de vraies images parfaites mais les proportions de la baseline, le FID CNN atteint déjà 77 % de celui de la baseline ([35]). Remettre les images de la baseline en proportion fait tomber son FID CNN de 162 à 31,6, soit −82 % de l'excès ([4]). RpGAN corrige justement ce mélange (écart des chiffres : 25,9 % → 4,9 %). Le FID CNN s'effondre donc. **[Vérifié]**
3. **À chiffre fixé, le CNN voit peu de progrès.** FID par chiffre : 58,2 → 53,5, soit −11 % de l'excès. Le reste du ÷18 vient du « 2 », rare et mal formé dans la baseline (199,6 → 114,8). **[Vérifié]**
4. **Inception voit surtout autre chose que le mélange des chiffres.** Les proportions de la baseline ne pèsent que 13 % de son excès ([95]). Remettre la baseline en proportion **aggrave** son FID Inception (+16 %, [72]), car on y fait entrer beaucoup de « 2 » mal formés. Inception réagit fortement au bruit de pixels ([75]). **Hypothèse :** il voit une texture ou un grain des images générées par le MLP que RpGAN + EMA réduit en partie. Le test « pixels gris » va dans ce sens mais ne le prouve pas (5,94 % → 5,66 %, réel 4,96 %).
5. **Les échelles diffèrent.** Un FID est une distance au carré dans l'unité des features du réseau : 128 nombres d'un côté, 2 048 de l'autre, d'amplitudes différentes. Les valeurs absolues ne se comparent pas ; seuls les rapports « au plancher » ou « à la baseline » ont un sens. En multiples du plancher à 10 000 images : la baseline est à environ 67× (CNN, 182/2,7) et environ 31× (Inception, 38,7/1,25 avec le plancher MNIST de Lucic 2018). RpGAN est à environ 3,7× (CNN) et environ 18× (Inception). **Selon le CNN, RpGAN est presque indiscernable de vraies images** : à N = 500, 21,0 contre un plancher de 18,9 ± 3,8, z = 0,6 [39]. **Selon Inception, il en est encore loin** (z = 184 [73]). **[Vérifié ; le plancher Inception à 10 000 images contre test n'a pas été mesuré par nous]**

**3.2 Pourquoi les deux FID choisissent des époques opposées** (ρ = −0,885 après l'époque 100). Après l'époque 100, la baseline améliore ses traits mais dérive vers un mélange de plus en plus déséquilibré : écart 15,2 % à l'époque 120, 27,3 % à l'époque 200 (`baseline.md:116-118`). Inception récompense les traits, le CNN pénalise le mélange. Cela confirme `baseline.md:60-75`. **[Vérifié]**

**3.3 Pourquoi le DRS « aide » en Inception mais pas en CNN sur RpGAN** (22,1 → 22,0 en Inception ; 9,9 → 14,2 en CNN). Le DRS change un peu le mélange des chiffres (écart 4,9 % → 5,9 %), et le CNN amplifie ce changement. **[Hypothèse cohérente avec §2.3, non testée directement]**

---

## 4. Ce qui marche / ne marche pas, et dans quel cas

**Arguments solides POUR garder un FID CNN (à mettre dans le rapport) :**
1. **Il est dans le domaine.** Inception n'a jamais vu de chiffre manuscrit ; le cours le dit (slide 9). Plusieurs papiers font ce remplacement sur MNIST (Bińkowski 2018 ; Hardy 2019 ; Dedeoglu 2021). **[Vérifié]**
2. **Il voit la perte de modes, Inception presque pas.** Un chiffre absent = 22-27 % de l'excès de la baseline en CNN, 0,9 % en Inception (z = 2,8 seulement). Pour un GAN **non conditionnel** sur 10 classes, perdre ou sous-produire un chiffre est le défaut typique : la baseline produit 1,6 % de « 2 ». **[Vérifié]**
3. **Il est bon marché.** Quelques secondes pour 10 000 images sur CPU, contre environ 40 min pour Inception sur ce CPU (environ 4 images/s mesuré, 2 threads). On peut donc le calculer à chaque époque. **[Vérifié]**
4. **Il est cohérent avec l'histogramme des chiffres et un FID par chiffre** (mêmes features), ce qui permet de **séparer mélange et qualité** (§2.3). Slide 83 du cours : *« Manifold uniformity is not the same as semantic class balance »*. **[Vérifié]**

**Limites honnêtes (à mettre aussi) :**
1. **Aveugle à ce qui ne sert pas à classer.** Le bruit sur 25 % des images ne pèse que 9-13 % de l'excès en CNN contre 102 % en Inception. Entre deux bons modèles, il ne tranche pas : ρ = 0,43 entre RpGAN et ses variantes, et à N = 500 RpGAN est indiscernable du réel. **Il sature quand les chiffres sont bien répartis.** **[Vérifié]**
2. **Dominé par le mélange des chiffres** (77-82 % du FID CNN de la baseline). Un tri ou un rééquilibrage a posteriori (DRS, sélection par classe) peut le faire baisser sans améliorer une seule image. C'est le « piratage » décrit par Kynkäänniemi 2023. **[Vérifié pour l'ampleur ; le piratage par le DRS reste une hypothèse]**
3. **Circularité.** La sélection d'instances est faite dans l'espace de ce CNN, puis évaluée dans ce même espace. La précision CNN monte en partie par construction (on garde les images proches du centre de chaque chiffre dans cet espace). Le sens du biais sur le FID est incertain. **Biais majeur pour le volet Données, à signaler.** **[Vérifié pour la circularité ; le sens du biais est une hypothèse]**
4. **Non comparable** aux papiers et au classement ; sa valeur dépend du CNN : 142 à 224 selon la seed du CNN (`baseline.md:87-93`). **[Vérifié]**
5. **Plus bruité.** Variation d'une seed d'entraînement à l'autre de 8,7-20 % contre 2,6-4,5 % pour Inception. À N = 500, son plancher varie de ±20 % d'un tirage à l'autre (Inception ±0,9 %), car chaque tirage change un peu le mélange des chiffres. **[Vérifié]**
6. **Il a vu `val`** (entraîné sur les 60 000 images de MNIST train) : plancher 1,68 contre val et 2,69 contre test. Ne jamais mélanger un chiffre val et un chiffre test. **[Vérifié]**
7. **Dimension et N : pas un problème pour le CNN** (128 dimensions, 10 000 images, 78 images par dimension, biais du plancher 2,9). **C'en est un pour Inception à petit N** : 2 048 dimensions, et à N = 500 la covariance n'est même pas de rang plein, d'où un plancher de 18,5. À 10 000 images (protocole du projet) le plancher Inception tombe vers 1,2. **[Vérifié]**
8. **Avant ou après la ReLU :** le choix « avant » est raisonnable (128 dimensions utiles, rang plein), mais la justification écrite (« FID instable ») n'est pas démontrée : après la ReLU le FID se calcule et garde l'ordre. **[Vérifié]**

**Inception, en miroir :** c'est la mesure du classement, elle est stable et comparable aux papiers, et elle voit le bruit, le grain et la qualité des traits. Mais elle est presque aveugle à un chiffre absent ou à des proportions fausses. Elle demande aussi d'agrandir 28 → 299 px (Parmar 2022 : le redimensionnement compte) et coûte cher sur CPU.

**Dans quel cas utiliser quoi :**

| Question | Outil |
|---|---|
| Classer les modèles, choisir un modèle ou une époque, chiffres finaux | **FID Inception** (+ P/R VGG), contre test, 10 000 images |
| Le GAN couvre-t-il les 10 chiffres, dans les bonnes proportions ? | Histogramme des chiffres + FID CNN |
| Les traits sont-ils meilleurs à chiffre fixé ? | FID CNN **par chiffre** (et idéalement Inception par chiffre) |
| Comparer à un papier ou au classement | Inception uniquement |
| Évaluer la sélection d'instances | **Pas le CNN seul** (circularité) : Inception + VGG |

---

## 5. Corrections des slides et des docs

1. **Slide « Two FIDs » / conclusion** (`comparaison.tex:77-82`, `:219-233`) : ajouter une ligne d'explication, par exemple *« Inception FID: the leaderboard metric. CNN FID: digit-aware, dominated by digit proportions (77 % of the baseline's CNN FID comes from its digit mix alone). »* Donner l'écart entre seeds : baseline CNN 182 ± 16 et Inception 38,7 ± 1,0.
2. **Conclusion :** les trois variantes RpGAN sont **à égalité en Inception** (22,1 / 22,0 / 21,4, ±0,5 à 1,0). Les écarts CNN (9,9 / 14,2 / 9,0, ±0,8 à 2,0) sont surtout du mélange de chiffres. Ne pas dire « + noise 0.3 is best » sur la base du CNN.
3. **« DRS only helps the baseline »** (`comparaison.tex:207`) : préciser *« neutral on Inception for RpGAN; worse on the CNN FID, which reacts to the digit mix »*.
4. **Choix de γ au FID CNN** (14,0 contre 17,7, ±0,9) : écart probablement réel, mais choisi avec la mesure qui distingue mal les bons modèles. À recontrôler en Inception, ou au moins le dire.
5. **Data : results** : les chiffres sont mesurés **uniquement** au CNN, dans l'espace où la sélection est faite. Ajouter *« selection and metric share the same CNN »*, et ne pas présenter « precision ↑ » comme un gain établi.
6. **`architecture-choix.md:171`** : « aucun choix n'est fait avec lui » est faux (sélection, γ). **`architecture-choix.md:172`** : remplacer « covariance singulière, FID instable » par « après la ReLU, 41 dimensions sur 128 sont toujours nulles (rang 87) : on perd de l'information ; avant la ReLU, les 128 dimensions sont utiles ».
7. **Ne jamais écrire « ÷18 » sans dire pourquoi :** c'est surtout la correction du mélange des chiffres, pas des images 18 fois meilleures. À chiffre fixé, le gain en FID CNN n'est que de 11 %.

---

## 6. Expériences

### 6.1 Faites (CPU local, `torch.set_num_threads(2)`)

| # | Expérience | Script | Sortie | Résultat principal |
|---|---|---|---|---|
| 1 | Plancher réel-vs-réel (train/val vs test), deux FID | `analyze.py` | `audit.csv` [2], [36], [70], [96]-[102] | CNN 10k : 2,69 (train), 2,93 (val) ; Inception 500 : 14,39 / 14,37 |
| 2 | Monotonie : bruit 0,1 et 0,3, flou, chimères, effacement, épaississement, à 10/25/50/100 % des images | `extract_features.py`, `analyze.py` | `audit.csv` [6]-[29], [40]-[63], [74]-[89] | monotone pour les deux ; Inception ≫ CNN sur le bruit |
| 3 | Perte de 1, 2, 5 chiffres ; déséquilibres (20 %, 30 % de « 1 », proportions de la baseline) | idem | [30]-[35], [64]-[69], [90]-[95] | CNN ×4 à ×25 plus sensible |
| 4 | Biais selon N (250 → 10 000) | idem | [103]-[119] | Inception 29,7 → 11,3 de 250 à 1 000 ; CNN 62,7 → 2,9 |
| 5 | Spearman sur toutes les paires existantes | `rank_correlation.py` | `rank_correlation.csv` | −0,885 (époques ≥ 100) ; 0,43 entre RpGAN |
| 6 | GAN réels : baseline, baseline remise en proportion, RpGAN (seed 0, `origin/noe/training:runs/cmploc_*`) | `extract_features.py`, `analyze.py` | [3]-[5], [37]-[39], [71]-[73] | proportions = 82 % du FID CNN, aggravent Inception |
| 7 | Décomposition et FID par chiffre (CNN) | `cnn_decomposition.py` | `cnn_decomposition.csv` | à chiffre fixé, −11 % seulement |
| 8 | Part de variance due au chiffre | `analyze.py` | `variance_share.csv` | CNN 0,67, Inception 0,26 |
| 9 | Avant / après ReLU | `analyze.py` | `relu.csv` | 41 dimensions nulles, ordre conservé |
| 10 | Pixels gris | `pixel_stats.py` | `pixel_stats.csv` | écart faible, ne tranche pas |

**Limites de ces expériences :** Inception à N = 500 seulement (biais fort, plancher 14,7) ; un seul tirage pour les défauts de pixels (les écarts-types viennent du plancher) ; un seul GAN par méthode (seed 0) ; les défauts artificiels ne sont que des approximations des défauts d'un GAN.

**Relancer** (depuis `wt-comparaison`, Python du venv, `DATA=<scratchpad>\data`) :
```
python <scratchpad>/scripts/fid/extract_features.py <scratchpad>      # ~25 min CPU (Inception), cache réutilisé
python <scratchpad>/scripts/fid/analyze.py <scratchpad>               # ~2 min
python <scratchpad>/scripts/fid/cnn_decomposition.py <scratchpad>     # secondes
python <scratchpad>/scripts/fid/rank_correlation.py <scratchpad>      # secondes
python <scratchpad>/scripts/fid/pixel_stats.py <scratchpad>           # secondes
python <scratchpad>/scripts/fid/make_figure.py <scratchpad>           # figure
```

### 6.2 À relancer (par priorité)

| Priorité | Expérience | Pourquoi | Coût | Commande |
|---|---|---|---|---|
| 1 | Mesurer les runs de sélection (`sel_95/90/75/50`, `sel_50_pixels`) en **Inception + VGG** | casser la circularité du volet Données | ~1 min par run et par seed sur GPU | `python evaluate.py --run runs/<sel_run> --weights last --seeds 0 1 2 --metric prof --reference test` (sur Juliet, runs de Nicodème) |
| 2 | Plancher Inception contre **test** à 10 000 images | normaliser proprement les chiffres finaux | ~1 min GPU (~80 min CPU) | `python evaluate.py --sanity --metric prof --reference test` |
| 3 | Refaire §2.3 à N = 10 000 pour Inception (défauts + GAN) | confirmer les ordres de grandeur trouvés à N = 500 | ~5 min GPU | mettre `N_POOL_HALF, N_DEG_INC, N_GAN_INC = 5000, 10000, 10000` dans `extract_features.py`, puis `analyze.py` avec `N_SMALL = 10000` |
| 4 | FID **par chiffre** Inception + CNN sur les 3 seeds des modèles finaux | séparer mélange et qualité dans le rapport | ~10 min GPU | adapter `cnn_decomposition.py` (même fonction `per_class_fid` avec `inception_features`) |
| 5 | Recontrôler γ (0,1 / 1 / 10) en Inception | le choix a été fait au CNN | ~3 min GPU par run | `python evaluate.py --run runs/<gamma_run> --weights last --seeds 0 --metric prof --reference test` |
| 6 | (option) KID ou FD-DINOv2 en troisième avis | estimateur sans biais (KID) / extracteur moderne (Stein 2023) | ~10 min GPU | `torchmetrics.image.kid.KernelInceptionDistance` ou `dgm-eval` (layer6six) |

---

## 7. Approfondissements possibles

- **FID par classe / Class-Aware Fréchet Distance** (Liu et al. 2018, arXiv 1803.07474) : ajuster une gaussienne **par chiffre** plutôt qu'une seule gaussienne. C'est exactement ce que notre FID par chiffre approche, et c'est la réponse la plus propre à « mélange contre qualité ».
- **KID** (Bińkowski 2018) : estimateur sans biais, donc comparable entre tailles d'échantillon ; utile si on mesure le DRS avec moins d'images.
- **Extracteur plus riche** (DINOv2, Stein 2023 ; CLIP + MMD = CMMD, Jayasumana 2024) : un troisième avis indépendant de notre CNN et des classes ImageNet.
- **Densité / couverture** (Naeem 2020) : version de la précision et du rappel moins sensible aux points aberrants, avec les mêmes features.
- **Montrer le « piratage »** dans le rapport : rééquilibrer a posteriori les sorties de la baseline baisse son FID CNN de 162 à 32 (−82 %) sans changer une seule image, et monte son FID Inception de 16 %. C'est un argument pédagogique fort pour la slide 86 (*« Explain what the measurements mean »*).

---

## 8. Sources

**Fichiers du projet (lecture seule)**
- `C:\Users\quent\Education\IASD\datalab\datalab 2\wt-comparaison\evaluate.py` (l. 3-5, 51-66, 115-121, 200-262, 273)
- `...\wt-comparaison\preprocessing.py` (l. 14, 19-22, 56)
- `...\wt-comparaison\architecture-choix.md` (l. 142-172, 206-207) ; `...\wt-comparaison\baseline.md` (l. 34, 60-104, 116-118, 178)
- `...\wt-comparaison\results\comparison_local.csv` ; `results\results.csv` ; `results\results_prof.csv` ; `runs\baseline_200ep_prof\metrics.jsonl`
- `C:\Users\quent\Education\IASD\datalab\datalab 2\masteriasd-datalab-project-a2-group-3\slides\comparaison.tex` (l. 57-63, 77-82, 207, 219-233)
- `C:\Users\quent\Education\IASD\datalab\datalab 2\enonce\diapos_enonce.pdf` (slides 8, 9, 33, 83, 86)
- `origin/nico/preprocessing:rapport-donnée-illustré.md` (l. 101-115, 313) ; générateurs `origin/noe/training:runs/cmploc_baseline_s0/G.pth` et `runs/cmploc_rpgan_r1_ema_s0/G.pth` (copiés dans `scratchpad\models\fid\`)

**Scripts et résultats de cet audit**
- Scripts : `C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad\scripts\fid\` (`extract_features.py`, `analyze.py`, `cnn_decomposition.py`, `rank_correlation.py`, `pixel_stats.py`, `make_figure.py`)
- Résultats : `C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad\results\fid\` (`audit.csv`, `rank_correlation.csv`, `cnn_decomposition.csv`, `variance_share.csv`, `relu.csv`, `pixel_stats.csv`, `fast_fid_check.csv`, `fid_sensitivity.png`, `cache\` = features)

**Papiers (vérifiés : PDF local ou page arXiv lue)**
- Heusel et al. 2017, *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (NeurIPS), PDF local — FID, fig. 3 (perturbations).
- Lucic et al. 2018, *Are GANs Created Equal?* (NeurIPS), PDF local — fig. 1 : plancher MNIST 1,25 ; sensibilité à la perte de modes.
- Chong & Forsyth 2020, *Effectively Unbiased FID and Inception Score and where to find them* (CVPR), PDF local.
- Kynkäänniemi et al. 2019, *Improved Precision and Recall Metric for Assessing Generative Models* (NeurIPS), PDF local.
- Kynkäänniemi et al. 2023, *The Role of ImageNet Classes in Fréchet Inception Distance* (ICLR) — https://arxiv.org/abs/2203.06026
- Stein et al. 2023, *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models* (NeurIPS) — https://arxiv.org/abs/2306.04675
- Jayasumana et al. 2024, *Rethinking FID: Towards a Better Evaluation Metric for Image Generation* — https://arxiv.org/abs/2401.09603 (version arXiv lue)
- Bińkowski et al. 2018, *Demystifying MMD GANs* (ICLR) — https://arxiv.org/abs/1801.01401 (§4 : LeNet pour MNIST, FID biaisé, KID sans biais)
- Parmar, Zhang, Zhu 2022, *On Aliased Resizing and Surprising Subtleties in GAN Evaluation* — https://arxiv.org/abs/2104.11222 (version arXiv lue)
- Naeem et al. 2020, *Reliable Fidelity and Diversity Metrics for Generative Models* (ICML) — https://arxiv.org/abs/2002.09797
- Hardy, Le Merrer, Sericola 2019, *MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets* — https://arxiv.org/abs/1811.03850 (classifieur MNIST pour le FID)
- Dedeoglu et al. 2021, *Continual Learning of Generative Models with Limited Data* — https://arxiv.org/abs/2101.09225 (annexe D, « modified FID » MNIST)
- Liu et al. 2018, *An Improved Evaluation Framework for Generative Adversarial Networks* — https://arxiv.org/abs/1803.07474 (CAFD, encodeur propre au domaine)
- Issenhuth et al. 2021, cité par le groupe (`architecture-choix.md:160-166`) ; non relu ici.

---

## Recommandation pour le rapport

**Quoi présenter :**
- **Mesure principale : FID Inception** + précision / rappel VGG, contre test, 10 000 images, 3 seeds, moyenne ± écart-type. C'est la métrique de la plateforme (slide 33 et page de résultats), la plus stable (2,6-4,5 % entre seeds), et elle est comparable aux papiers.
- **Mesure de diagnostic : FID CNN**, toujours présenté **avec l'histogramme des chiffres**, et si possible avec un **FID par chiffre** pour séparer « bon mélange » et « bons traits ».
- **Ne pas** classer les variantes RpGAN avec le FID CNN. **Ne pas** évaluer la sélection d'instances avec le seul CNN.

**Phrase de justification (français) :**
> Nous rapportons deux FID. Le FID Inception est la métrique de la plateforme ; il est comparable aux travaux publiés et réagit surtout à l'aspect des images (bruit, traits). Inception n'ayant jamais vu de chiffres, il est presque aveugle à un chiffre manquant (+2 % de FID quand un chiffre sur dix disparaît, contre +181 % pour le FID CNN ; mêmes 500 images). Nous ajoutons donc un FID calculé sur les features d'un petit CNN entraîné sur MNIST, comme Bińkowski et al. (2018) : 67 % de la variance de ses features est due au chiffre, et il voit la perte et le déséquilibre des chiffres, défaut typique d'un GAN non conditionnel, mais il reste peu sensible au bruit.

**English version (for the slides/report):**
> We report two FIDs. Inception FID is the leaderboard metric, comparable to published work; it mostly reacts to image appearance (noise, strokes) but is nearly blind to a missing digit (+2 % when one digit in ten is dropped, vs +181 % for the CNN FID; same 500 images). We add a FID on the features of a small MNIST-trained CNN, as in Bińkowski et al. (2018): 67 % of its feature variance is digit identity, so it detects missing or misproportioned digits, the typical failure of an unconditional GAN, but is weak on noise.

**Figure clé :** `fid_sensitivity.png`. Pour chaque défaut dosé appliqué à de vraies images (et pour les deux GAN), le poids du défaut dans chaque FID, en % de l'écart entre la baseline et de vraies images. Elle montre d'un coup d'œil : bruit → Inception ; chiffres manquants ou mal répartis → CNN ; RpGAN « parfait » pour le CNN mais encore à 56 % pour Inception.
