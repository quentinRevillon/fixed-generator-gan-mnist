# Volet Entraînement (Noé) : RpGAN + R1 + EMA — analyse pour le rapport final

*Préparé le 8/11/2026, en lecture seule (dépôt et worktree non modifiés). Les branches lues : `origin/noe/training` (edf14fb, code avec EMA, logs, poids), `origin/main` (fbdd93f, modèle rendu, slides présentées), worktree `wt-comparaison` (`quentin/comparaison`, 7474210).*

**Légende.** **[V]** = vérifié (fichier, log ou calcul refait). **[H]** = hypothèse (raisonnement plausible, pas testé). **[L]** = lu dans un article.

## 0. Résumé en 10 lignes

1. **La baseline du protocole commun n'a PAS d'EMA** [V] (`configs/compare_local/baseline.yaml` : aucune section `ema`; log `cmploc_baseline_s0.log:1` : « EMA non »). La comparaison commune 38,7/182 → 22,1/9,9 mélange donc **perte + R1 + EMA**.
2. **Nouvelle mesure (faite ici, sans réentraîner)** : avec EMA, `train.py` n'utilise jamais le G moyenné pendant l'entraînement, donc `G_raw.pth` = le G d'un run sans EMA. J'ai mesuré `G.pth` (EMA) et `G_raw.pth` (brut) de tous les runs (petit CNN, contre MNIST test). Mes chiffres reproduisent exactement `results/comparison_local.csv` (9,93 = 9,93) [V].
3. **Protocole commun, FID du petit CNN** : baseline (brute) 181,8 ± 19,4 → **RpGAN + R1 brut 19,7 ± 4,2** → RpGAN + R1 + EMA 9,9 ± 2,5 [V]. La perte + R1 divise le FID par ≈ 9 ; l'EMA le divise encore par ≈ 2. **En FID Inception (seed 1, 2000 images), c'est l'inverse : baseline 46,1, RpGAN + R1 brut 47,3, + EMA 30,2** [V, mesure partielle] : sans EMA, la perte ne gagne rien en Inception ; tout le gain Inception viendrait de l'EMA. À confirmer sur Juliet (P1a, 10 min).
4. **Campagne 100 époques (60k images)** : BCE brut 70,0 → BCE + EMA 63,2 (−10 %) ; RpGAN + R1 brut 27,2 → + EMA 13,3 (−51 %) [V]. **L'EMA aide beaucoup RpGAN, presque pas la BCE** : il y a une interaction.
5. **Les bonnes proportions de chiffres viennent de la perte, pas de l'EMA** : écart-type des proportions 3,3 points (BCE, brut ou EMA) contre 1,4-1,6 (RpGAN + R1, brut ou EMA) [V].
6. **Combinaisons jamais mesurées** : BCE + R1 (le code ne le permet pas : `r1_gamma` est ignoré avec `bce`), baseline + EMA au protocole commun, RpGAN sans R1 au-delà de 10 époques / 1 seed. Sans BCE + R1, on ne peut pas séparer « perte relativiste » et « R1 ».
7. **Divergence avec R3GAN** [L] : dans le papier cité, **RpGAN + R1 sans R2 échoue** sur StackedMNIST (« Fail », γ de 0,1 à 100). Chez nous, RpGAN + R1 marche et R2 n'apporte rien. À dire à l'oral et dans le rapport.
8. **γ = 1 est exactement le γ « critique » de Mescheder (2|f′(0)| = 1 pour la perte logistique)** [L + calcul], ce qui explique bien γ = 0,1 (oscillations peu amorties → divergence) et γ = 10 (convergence lente) [H].
9. **Le D utilisé par le DRS a été entraîné contre le G brut, pas contre G_EMA** [V] (`train.py:77-80`), et c'est un D relativiste : deux raisons pour que le DRS n'aide pas RpGAN + R1 + EMA (CNN 9,9 → 14,2).
10. Slides : formules justes ; à corriger : attribution (slide 1), « RpGAN: none » (slide 12), « The loss improves both FIDs » et « The loss gives the largest gain » (slides 19 et 21 : la baseline n'a pas d'EMA, et en Inception le gain semble venir de l'EMA). Le code du modèle rendu (RpGAN + EMA) n'est **pas sur `main`** [V].

---

## 1. Attendu (littérature)

### 1.1 Définitions

- **Discriminateur D** : réseau qui donne un **score brut** `D(x)` (un « logit », avant la sigmoïde) ; plus il est grand, plus D trouve l'image réaliste.
- **Perte BCE de la baseline** (GAN « non saturant ») : D minimise `−log σ(D(réel)) − log(1 − σ(D(faux)))`, G minimise `−log σ(D(faux))` (σ = sigmoïde).
- **RpGAN** (*relativistic pairing GAN*, Jolicoeur-Martineau 2019, appelé RSGAN dans son article) : D juge une **paire** (une vraie, une fausse) : D minimise `softplus(D(faux) − D(réel))`, G minimise `softplus(D(réel) − D(faux))`. `softplus(t) = log(1 + eᵗ)`. Seule la **différence** des scores compte : ajouter une constante à D ne change rien.
- **R1** (Mescheder et al. 2018) : pénalité `γ/2 · ‖∇ₓD(x)‖²` sur les **vraies** images : elle interdit à D d'avoir une pente forte autour des données. **R2** : même pénalité sur les **fausses** images. **γ** règle la force de la pénalité.
- **EMA** (*exponential moving average*) du générateur : on garde une copie `G_ema ← β·G_ema + (1−β)·G` après chaque pas ; on évalue et on rend G_ema. β = 0,999 ici. **Demi-vie** = nombre de pas pour que le poids d'un pas ancien soit divisé par 2 : ln 0,5 / ln 0,999 ≈ **693 pas** ≈ 69 000 images ≈ 1,2 époque (60k) ou 1,4 époque (50k) [calcul].
- **Convergence locale** : si on démarre près de l'équilibre (G = données), l'entraînement y reste et s'en rapproche.
- **Mode** : ici, un chiffre (0 à 9). **Oubli de modes** (*mode dropping*) : G ne produit plus certains chiffres ; **effondrement** (*collapse*) : G produit une seule image.

### 1.2 Ce que disent les articles

| Article | Résultat chiffré | Ce qu'on en attend chez nous |
|---|---|---|
| **Huang et al. 2024, R3GAN** (NeurIPS 2024, arXiv 2501.05441) [L] | StackedMNIST (1000 modes) : **RpGAN + R1 + R2 : 1000 modes, KL 0,0781** ; GAN + R1 + R2 : 693 modes, KL 0,9270 ; **RpGAN + R1 : « Fail »**, GAN + R1 : « Fail » ; note 2 : « faire varier γ de 0,1 à 100 ne stabilise pas » (Table 3 de l'article, citée « Tab. 2 » dans le texte, §2.4). Théorie (s'appuie sur Sun et al. 2020) : le paysage de RpGAN « ne contient pas de minimum local correspondant à un oubli de modes ». Prop. II : RpGAN **avec R1 ou R2** est localement convergent ; mais « R1 seul ne suffit pas pour un entraînement globalement convergent ». | RpGAN couvre tous les chiffres ; R2 nécessaire en pratique. |
| R3GAN, hyperparamètres StackedMNIST (Table 19) [L] | γ **1 → 0,1** (décroissance cosinus), lr 2e-4 constant, batch 512, Adam β₂ 0,9 → 0,99 (β₁ = 0, §3 config B), EMA demi-vie 0 → 0,5 M images, 10 M images vues, G ≈ 20,7 M paramètres (ResNet). | Notre γ = 1 = leur valeur de départ ; notre EMA a une demi-vie ≈ 7× plus courte (0,069 M images) ; notre protocole commun voit 10 M images aussi (100 000 pas × 100). |
| R3GAN, « roadmap » FFHQ-256 (Table 4) [L] | B (StyleGAN2 épuré, **déjà avec R1**) 12,46 → C (RpGAN) 11,77 → + R2 11,65 → D (ResNet) 9,95 → E (convolutions groupées, goulot inversé) **7,05**. | Chez eux, **la perte apporte ≈ 0,8 point, l'architecture ≈ 4,6**. Notre architecture est imposée : on ne teste que la petite partie « perte ». |
| **Jolicoeur-Martineau 2019** (ICLR) [L] | CIFAR-10, 100k itérations, 1 seed. Réglage stable (lr 2e-4, β (0,5 ; 0,999)) : SGAN 40,64 → **RSGAN 36,61** → RaSGAN 31,98 ; RSGAN + pénalité de gradient 25,60 (Table 2). Réglage instable lr 1e-3 : SGAN 154,20 → **RSGAN 50,95** (Table 3). | Gain modeste quand la baseline est stable, **gain fort quand elle est instable**. Notre baseline est instable (1 seed sur 3 s'effondre). |
| **Mescheder et al. 2018** [L] | Dirac-GAN : sans régularisation, l'entraînement **tourne en rond** autour de l'équilibre. Lemme 3.3 : avec pénalité de gradient, valeurs propres `λ = −γ/2 ± √(γ²/4 − f′(0)²)` ; **γ critique = 2|f′(0)|** (pas de rotation). Théorème 4.1 : R1 ou R2 → convergence locale (petit lr). Images : sans régularisation, « effondrement rapide » ; R1 stabilise. Leurs expériences : γ = 10. | Avec la perte logistique, f(t) = −log(1+e⁻ᵗ), f′(0) = 1/2 → **γ critique = 1** [calcul]. |
| **Yazıcı et al. 2019** (ICLR) [L] | L'EMA converge vers des cycles d'amplitude qui tend vers 0 autour de l'équilibre (jeux bilinéaires). CIFAR-10 (architecture conventionnelle) : FID **16,84 → 12,56** avec EMA β = 0,9999 (Table 2, 3 runs). Effet de β (Table 4, CIFAR-10, 4 runs) : sans moyenne 18,14 ; β = 0,999 : **13,77** ; β = 0,9999 : 12,91. | EMA : **−11 % à −29 % de FID** ; ne corrige pas un effondrement (rien ne le dit, et G_ema n'influence pas l'entraînement). |

---

## 2. Observé chez nous

### 2.1 Vérification des chiffres des slides et du rapport de Noé [V]

Script : `scratchpad/scripts/training/log_stats.py` (sortie `log_stats.out`), moyenne ± écart-type (ddof = 1) sur les seeds.

| Chiffre affiché | Recalcul | Source | Verdict |
|---|---|---|---|
| γ = 1 : FID 14,0, ± 0,9 | 14,03 ± 0,88 (13,07 ; 14,80 ; 14,23) | `results.csv:13-15` | ✔ |
| γ = 10 : FID 17,7 | 17,69 ± 0,88 | `results.csv:19-21` | ✔ (± 0,9 aussi) |
| R3GAN (R1 + R2) : 14,4 ± 0,9 | 14,42 ± 0,94 | `results.csv:16-18` | ✔ |
| BCE : 58,7 ± 4,8 | 58,66 ± 4,82 **sur 2 seeds** (62,07 ; 55,25) | `results.csv:10,12` | ✔ mais 2 seeds, pas 3 |
| Seed BCE effondrée : 2770 | 2770,19, 100 % de « 8 » | `results.csv:11` | ✔ |
| γ = 0,1 diverge | FID 2091-2672 aux époques 10 et 20, arrêt à l'époque 27-28 | `logs/ema100_rpgan_r1_g0.1_s*.log` | ✔ (FID > 2000 à l'époque **20**, pas 27) |
| Commun : baseline 38,7 / 182 | 38,66 ± 1,23 / 181,8 ± 19,4 | `results/comparison_local.csv:2,4,6` | ✔ |
| Commun : RpGAN+R1+EMA 22,1 / 9,9 | 22,14 ± 1,26 / 9,93 ± 2,47 | `comparison_local.csv:14,16,18` | ✔ |
| Commun : + DRS 22,0 / 14,2 ; + bruit 21,4 / 9,0 ; baseline + DRS 34,4 / 154 | 22,04 / 14,17 ; 21,44 / 9,01 ; 34,36 / 153,96 | idem | ✔ |
| Modèle rendu « seed 1 » | `checkpoints/G.pth` de `main` = `runs/cmploc_rpgan_r1_ema_s1/G.pth` (même hash git 66e370e) ; CNN FID test 7,60, Inception 21,15 | commit fbdd93f | ✔ (seed choisie sur le FID CNN de **val** parmi 3 : c'est le meilleur des 3, la moyenne est 9,9) |
| Proportions 5.2 (BCE 4,2 % / 14,7-15,9 %, sd 3,0-3,4 ; RpGAN 7,7-8,0 % / 12,0-12,6 %, sd 1,2-1,6) | identiques | `results.csv` (colonne classes) | ✔ |

Remarque : la colonne `seed` de `comparison_local.csv` vaut toujours 0 : c'est la seed du tirage de z à la mesure, pas la seed d'entraînement (`scripts/measure_runs.py:18`). Le nom du run porte la seed d'entraînement.

### 2.2 Ablation EMA refaite ici, sans réentraînement [V]

**Pourquoi c'est possible.** Dans `train.py` (branche `noe/training`), G_ema n'est mis à jour qu'après le pas de G (`train.py:79-80`) et ne sert qu'à l'évaluation et à la sauvegarde (`train.py:130,161,176,183`). Le commentaire `train.py:127` précise qu'il n'y a aucun tirage aléatoire en plus. Donc la trajectoire du G brut d'un run « avec EMA » est **exactement** celle du même run sans EMA (même machine). `G_raw.pth` est versionné pour tous les runs `ema100_*` et `cmploc_rpgan_*`.

**Mesure.** `scratchpad/scripts/training/ema_ablation.py` : 10 000 images, z de la seed 0, petit CNN, contre MNIST test (même code `evaluate.evaluate` que `measure_runs.py`). Sorties : `ema_ablation.csv`, `summarize_ablation.out`. Contrôle : mes `G.pth` du protocole commun donnent 12,52 / 7,60 / 9,69, **identiques** à `comparison_local.csv` (12,517 / 7,597 / 9,685). Pour la campagne 100 époques, ma mesure diffère un peu de `results.csv` (13,25 contre 14,03 pour RpGAN + R1 : les runs datent d'avant le changement d'`evaluate.py`), donc **je ne compare G et G_raw qu'avec ma mesure**.

| Config (CNN FID, test) | G brut (sans EMA) | G_EMA | Effet de l'EMA | Precision brut → EMA | Recall brut → EMA | sd proportions (pts) brut / EMA |
|---|---|---|---|---|---|---|
| **100 ép., BCE** (2 seeds stables) | 70,0 ± 17,9 (82,7 ; 57,4) | 63,2 ± 5,5 | **−10 %** (s1 : +3 %) | 0,76 → 0,79 | 0,71 → 0,79 | 3,34 / 3,36 |
| BCE seed 2 (effondrée) | 2770 | 2771 | aucun | — | 0 → 0 | 30 / 30 |
| **100 ép., RpGAN + R1 γ = 1** | 27,2 ± 2,7 | 13,3 ± 0,9 | **−51 %** | 0,70 → 0,72 | 0,73 → 0,80 | 1,60 / 1,42 |
| 100 ép., R3GAN (R1 + R2) | 30,5 ± 6,1 | 15,7 ± 0,2 | −49 % | 0,69 → 0,70 | 0,72 → 0,80 | 1,79 / 1,50 |
| 100 ép., RpGAN + R1 γ = 10 | 36,3 ± 8,2 | 18,4 ± 0,9 | −49 % | 0,65 → 0,67 | 0,72 → 0,79 | 1,85 / 1,70 |
| **Commun 200 ép., baseline** (pas d'EMA) | **181,8 ± 19,4** | *non mesuré* | ? | 0,83 | 0,75 | — |
| **Commun 200 ép., RpGAN + R1** | **19,7 ± 4,2** (22,9 ; 15,0 ; 21,3) | 9,9 ± 2,5 | **−50 %** | 0,71 → 0,74 | 0,77 → 0,83 | 1,43 / 1,27 |
| Commun, RpGAN + R1 + bruit 0,3 | 18,3 ± 2,1 | 9,0 ± 1,0 | −51 % | 0,71 → 0,74 | 0,78 → 0,82 | 1,22 / 1,18 |

**Lecture.**
- **À 100 époques (EMA des deux côtés)**, la perte divise le FID par 63,2 / 13,3 ≈ **4,8** ; **sans EMA des deux côtés**, par 70,0 / 27,2 ≈ **2,6**. Le facteur 4 du rapport de Noé (58,7 → 14,0) est donc obtenu *avec* l'aide de l'EMA, qui profite surtout à RpGAN.
- **Au protocole commun**, comparer à perte égale d'EMA (les deux bruts) : 181,8 → 19,7, ÷ 9,2. L'EMA fait ensuite 19,7 → 9,9. **La conclusion « la perte donne le plus grand gain » reste vraie en FID CNN**, mais **la moitié du FID final de RpGAN vient de l'EMA**, et la baseline n'a pas eu cette chance. **En FID Inception, la conclusion s'inverse probablement** (§2.5). L'effet de l'EMA sur la baseline à 200 époques n'est pas mesuré ; à 100 époques il n'était que de −10 % [V], donc la baseline + EMA resterait très probablement > 100 en FID CNN [H].
- **L'EMA ne change pas les proportions** (sd 3,34 → 3,36 pour la BCE ; 1,60 → 1,42 pour RpGAN) : **l'équilibre des chiffres vient de la perte RpGAN + R1** [V]. L'EMA améliore surtout le **recall** (+0,06 à +0,08) et un peu la precision.
- **R2 n'aide pas**, avec ou sans EMA (30,5 contre 27,2 brut ; 15,7 contre 13,3 EMA) [V].
- FID Inception de G brut au protocole commun : voir 2.5.

### 2.3 Les courbes dans le temps [V] (`log_stats.out`, une mesure toutes les 10 époques, CNN contre val pour `cmploc`, contre test pour `ema100`)

- **BCE, démarrage lent** : FID encore 600-905 à l'époque 30 et recall = 0 jusqu'à l'époque 20 (0,005 à l'époque 30) (`logs/baseline_rerun.log`, `ema100_baseline_s*.log`). RpGAN + R1 : FID 186-193 dès l'époque 10, 57-61 à l'époque 20.
- **Effondrement BCE (ema100, seed 2)** : dès l'**époque 7**, `D(x) = 1,00`, `D(G(z)) = 0,00`, `loss_G = 100,000` jusqu'à la fin (`logs/ema100_baseline_s2.log`). 100,000 est la borne de `nn.BCELoss` (log tronqué à −100) : σ(D(G(z))) vaut exactement 0 en float32 pour toutes les images. Alors la dérivée de la sigmoïde σ(1−σ) = 0 et **G ne reçoit plus aucun gradient** : l'effondrement est **définitif**. Mécanisme : vérifié dans le log ; le fait que `BCELoss` tronque à −100 est documenté par PyTorch.
- **Baseline commune sans EMA : elle se dégrade après l'époque 100-120** (FID CNN val) : meilleur 50,0 (ép. 120), 64,8 (ép. 100), 63,1 (ép. 110), puis **159 / 196 / 179 à l'époque 200** ; la precision monte (0,85) et le recall baisse (0,75). À l'époque 200, les « 1 » font 24-27 % et les « 2 » 1-2 % (`comparison_local.csv:2,4,6`, colonne classes). **Le 182 du protocole commun mesure donc la baseline dans sa phase dégradée** : le choix « époque 200 » vient du FID Inception (`baseline.md`), le FID CNN aurait choisi l'époque ≈ 110.
- Avec EMA, la baseline montre le même début de dérive à 100 époques : 58,7 (ép. 80) → 62,1 (ép. 100) pour s0, 51,7 (ép. 90) → 55,2 pour s1 (`logs/ema100_baseline_s0.log`, `_s1.log`).
- **RpGAN + R1 + EMA baisse jusqu'au bout** : 100 époques, encore −0,1 à −1,0 entre les époques 90 et 100 ; 200 époques, meilleur à l'époque 170-200 pour les 3 seeds (`cmploc_rpgan_r1_ema_s*.log`). **Pas convergé** : un entraînement plus long aiderait probablement [H].
- **Équilibre en fin d'entraînement** : RpGAN + R1 : σ(D(x)) ≈ 0,62, σ(D(G(z))) ≈ 0,59 (écart de logits ≈ 0,13) ; BCE : 0,67 / 0,33 (ema100), 0,72 / 0,28 (commun). Le D de la baseline sépare encore nettement vrai et faux.
- **RpGAN sans pénalité (γ = 0)** : dès l'époque 2, `loss_D = 0,000` et `loss_G = 28 à 55` (= écart de logits entre vrai et faux, car softplus(t) ≈ t) ; FID 2619 à l'époque 10, 100 % de « 8 » (`logs/train_rpgan.log`). Une seule seed, 10 époques.
- **γ = 0,1** : dès l'époque 2-3, `loss_D ≈ 0,005`, `loss_G ≈ 6,7` (écart de logits ≈ 6,7) ; FID > 2000 aux époques 10 et 20 ; seed 2 : precision 0,94, recall 0 (une seule forme réaliste) (`logs/ema100_rpgan_r1_g0.1_s*.log`).
- **R3GAN + Adam (0 ; 0,99)** : FID 1430-1560 de l'époque 10 à 50 (`logs/train_r3gan_adam.log`).

### 2.4 Coût [V]

- Mac M3 (MPS), runs en parallèle : 44 s/ép. BCE, 46 s R1, 48 s R1 + R2 (`training-r3gan.md:106`) ; campagne 100 époques : ≈ 7 h 40 par run à 6 en parallèle (`logs/sweep_training.log`) ; protocole commun : 2 h 20 - 2 h 45 par run de 200 époques à 4-5 en parallèle (`logs/compare_local.log`).
- A100 (Juliet) : baseline 200 époques ≈ **14 min d'entraînement seul**, 49 min avec une mesure Inception + VGG toutes les 2 époques (≈ 20 s par mesure) (`baseline.md:189`).

### 2.5 FID Inception de G brut (protocole commun) — mesure partielle, 1 seed, 2000 images [V]

Script `ema_inception.py` → `ema_inception.csv`. Inception sur CPU est lent (≈ 6 min pour 2000 images ici) : je n'ai pu mesurer que la **seed 1** et **2000 images** (pas 10 000). Le FID à 2000 images est biaisé vers le haut ; on ne compare donc qu'entre lignes à 2000 images, toutes avec les **mêmes** 2000 images réelles de MNIST test.

| Seed 1, protocole commun | FID Inception, 2000 images | FID Inception, 10 000 images (`comparison_local.csv`) |
|---|---|---|
| Baseline (sans EMA) | **46,1** | 37,4 |
| RpGAN + R1, G brut (sans EMA) | **47,3** | non mesuré (≈ 38-39 si le biais est le même, ≈ +9 [H]) |
| RpGAN + R1 + EMA (= modèle rendu) | **30,2** | 21,2 |

**Lecture.** Pour cette seed, **sans EMA, RpGAN + R1 n'est pas meilleur que la baseline en FID Inception** (47,3 contre 46,1) ; **tout le gain Inception (≈ −16 points) vient de l'EMA**. En FID CNN, au contraire, l'essentiel vient de la perte (182 → 19,7 brut). Interprétation [H] : le FID CNN (features d'un classifieur de chiffres) est dominé par les **proportions de chiffres**, que corrige la perte ; le FID Inception est plus sensible à la **qualité de chaque image** (grain, artefacts : Noé note des « artefacts granuleux », `training-r3gan.md:147`), que l'EMA lisse en moyennant les poids. **À confirmer sur 3 seeds et 10 000 images (P1a, §6, 10 min sur Juliet)** avant de l'écrire dans le rapport.

### 2.6 Matrice d'ablation {perte} × {R1} × {EMA} : ce qui est mesuré, ce qui manque [V]

« Brut (ici) » = mesuré dans cette analyse via `G_raw.pth` (pas besoin de réentraîner). CNN = FID du petit CNN contre test.

| Perte | R1 | EMA | Campagne 100 ép. (60k) | Protocole commun 200 ép. (50k) |
|---|---|---|---|---|
| BCE (sigmoïde + `BCELoss`) | non | non | ✔ brut (ici) : 70,0 (2/3 stables) ; 50 ép. : 188 (1 seed) | ✔ 181,8 CNN / 38,7 Inception (3 seeds) |
| BCE | non | oui | ✔ 58,7 (`results.csv`) / 63,2 (ici), 2/3 stables | **✗ manque** (P1b) |
| BCE | oui | non / oui | **✗ jamais** (le code ignore `r1_gamma` avec `bce`, `utils.py:79-83`) | **✗** (P2) |
| RpGAN | non | non | ✔ 10 ép., **1 seed** : diverge (2619) | ✗ (inutile : diverge) |
| RpGAN | non | oui | ✗ (EMA d'un G effondré : sans intérêt) | ✗ |
| RpGAN | oui (γ = 1) | non | ✔ brut (ici) : 27,2 ± 2,7 ; 50 ép. : 43,6 (1 seed) | ✔ brut (ici) : 19,7 ± 4,2 CNN ; Inception seed 1 seulement (§2.5) |
| RpGAN | oui (γ = 1) | oui | ✔ 14,0 ± 0,9 | ✔ 9,9 CNN / 22,1 Inception |
| RpGAN | oui (γ = 0,1 / 10) | oui / non | ✔ diverge / 17,7 (EMA), 36,3 (brut, ici) | ✗ |
| RpGAN | R1 + R2 | oui / non | ✔ 14,4 (EMA), 30,5 (brut, ici) | ✗ (`configs/compare/r3gan.yaml` existe, jamais lancé) |

**Ce qui empêche d'attribuer le gain :** (1) **BCE + R1** n'existe pas : on ne peut pas séparer « relativité » et « R1 » ; (2) **baseline + EMA au protocole commun** n'existe pas : on ne sait pas combien l'EMA aurait apporté à la baseline (en FID Inception surtout) ; (3) RpGAN sans R1 : 1 seed, 10 époques.

**Le D du DRS** : `D.pth` est le D de la dernière époque, entraîné contre le **G brut** (`train.py:77-78` : `D_train(x, G, …)` avec le G brut ; G_ema n'apparaît qu'aux lignes 79-80, 130, 161, 176, 183). Le DRS du protocole commun applique ce D à **G_EMA** (`scripts/compare_local.py:81-82` : `G.pth` + `D.pth`). [V]

---

## 3. Divergences avec la littérature, et pourquoi

| # | Attendu | Observé | Pourquoi (statut) |
|---|---|---|---|
| D1 | R3GAN : **RpGAN + R1 échoue** sur StackedMNIST, R2 nécessaire | RpGAN + R1 marche (3/3 seeds, + 3/3 au protocole commun) ; R2 n'apporte rien (14,4 contre 14,0) | [H] Tâche plus facile (10 modes, 1 canal 28×28, contre 1000 modes, 3 canaux 32×32) ; réseaux minuscules (G 1,49 M, D 1,46 M paramètres [calcul depuis `model.py`] contre ≈ 20,7 M) ; Adam β₁ = 0,9 chez nous contre β₁ = 0 chez eux (avec leurs β, notre run s'effondre : `results.csv:8`) ; batch 100 contre 512. Le papier dit lui-même que la preuve locale marche avec R1 **ou** R2 (Prop. II) : R2 est un choix empirique, pas une nécessité théorique. |
| D2 | R3GAN : la perte apporte peu (FFHQ 12,46 → 11,65), l'architecture beaucoup | Chez nous, la perte (+ R1) apporte énormément (÷ 2,6 à ÷ 9 en FID CNN, selon le protocole) | [V/H] Leur point de départ a **déjà R1** ; le nôtre n'a aucune régularisation et une BCE calculée **après** la sigmoïde, qui peut tuer le gradient de G (2.3). Une partie du gain pourrait venir de « logits + R1 » plutôt que de la relativité : **non testé** (BCE + R1 manquant, §6). |
| D2b | R3GAN : la perte améliore le FID Inception (FFHQ, CIFAR) | Protocole commun, seed 1 : RpGAN + R1 **brut** 47,3 contre baseline 46,1 (Inception, 2000 images) ; le gain Inception n'apparaît qu'avec l'EMA (30,2) | [V partiel, H pour la cause] Notre gain « perte » porte surtout sur les proportions (que le FID CNN voit très fort) ; la qualité par image, que voit Inception, dépend de l'EMA et de l'architecture. |
| D3 | Yazıcı : EMA −11 % à −29 % | RpGAN + R1 : **−50 %** ; BCE : **−10 %** | [H] RpGAN + R1 tourne près d'un équilibre (D(x) ≈ D(G(z))) : G oscille autour d'une bonne solution et la moyenne tombe près du centre (mécanisme de Yazıcı). La BCE a une erreur **systématique** (mauvaises proportions, dérive lente sur des dizaines d'époques) que l'EMA (fenêtre ≈ 1,2 époque) ne peut pas corriger. |
| D4 | Jolicoeur-Martineau : gain relativiste modeste en réglage stable, fort en réglage instable | Fort | [L/V] Cohérent : notre baseline est instable (1/3 effondrée à 100 ép. ; 15/34 dans le volet Données, slide 9). |
| D5 | Mescheder : γ critique = 2\|f′(0)\| | γ = 1 meilleur ; 0,1 diverge ; 10 moins bon | [L + calcul, application H] f′(0) = 1/2 → γ critique = 1. Valeurs propres du Dirac-GAN : γ = 0,1 → −0,05 ± 0,50 i (rotation presque pas amortie ; avec des pas finis d'Adam, la rotation peut sortir du domaine stable) ; γ = 1 → −0,5 (double, convergence la plus rapide, sans rotation) ; γ = 10 → −0,025 et −9,975 (une direction converge 20× plus lentement qu'à γ = 1). L'échelle de γ dépend aussi de la norme des images (784 pixels dans [−1, 1]) : la coïncidence « γ = 1 » est suggestive, pas démontrée. |
| D6 | « Images individuelles plus nettes » (attente naïve d'un meilleur FID) | Precision CNN plus basse (0,72 contre 0,80 à 100 ép. ; 0,74 contre 0,83 au commun) | [V/H] Le FID CNN baisse surtout par **les proportions** et la **couverture**. La BCE se concentre sur les chiffres faciles (« 1 », « 7 ») ⇒ precision haute. R1 lisse D près des données ⇒ gradient moins « piqué » vers les détails fins [H]. G entièrement connecté, sortie tanh, sans convolution : la netteté est limitée par l'architecture imposée [H]. |
| D7 | DRS (Azadi) suppose D ≈ log(p_data / p_G) | DRS n'aide pas RpGAN + R1 + EMA (CNN 9,9 → 14,2 ; Inception 22,1 → 22,0) | [V] Le D sauvegardé a été entraîné contre le **G brut** (`train.py:77-78`), pas contre G_EMA. [L/H] Un D relativiste n'estime qu'une **différence** de scores (le niveau absolu est libre), pas un rapport de densités (`training-r3gan.md:161` le note déjà). |

---

## 4. Ce qui marche, ce qui ne marche pas, et dans quel cas

### 4.1 Ce qui marche

- **RpGAN + R1 (γ = 1)** : 0 effondrement sur 6 runs (3 à 100 ép., 3 au commun) ; aussi 0/3 avec R2 et 0/3 à γ = 10 [V]. Bonnes proportions (sd 1,2-1,6 point, contre 0,73 pour du vrai MNIST et 3,3 pour la BCE) [V].
  - *Mécanisme (perte)* [L/H] : comparer une vraie et une fausse image donne à G un gradient tant qu'il existe un écart, même quand D est sûr de lui (softplus sur les logits ne sature jamais) ; la théorie (Sun et al. 2020, reprise par R3GAN) dit qu'il n'y a pas d'équilibre local « avec un mode oublié ». Le lien précis avec les **proportions** est notre interprétation, pas un théorème.
  - *Mécanisme (R1)* [L] : R1 empêche D de devenir raide autour des données ; près de l'équilibre, cela supprime la rotation de l'entraînement (Mescheder, Lemme 3.3) ⇒ convergence locale.
- **EMA** sur RpGAN + R1 : FID CNN ÷ 2 à coût nul, sur toutes les configs RpGAN (−49 à −51 %) [V] ; FID Inception 47,3 → 30,2 (seed 1, 2000 images) [V partiel]. *Mécanisme* [L] : moyenne des oscillations autour de l'équilibre (Yazıcı).
- **Bruit d'instance 0,3 + RpGAN + R1 + EMA** : 9,0 ± 1,0 contre 9,9 ± 2,5 (CNN), 21,4 contre 22,1 (Inception) : petit gain, dans le bruit des seeds [V].

### 4.2 Ce qui ne marche pas

- **RpGAN sans R1 (γ = 0)** : D gagne dès l'époque 1-2 (écart de logits 28-55), effondrement [V, 1 seed]. Sans pénalité, rien n'empêche D de séparer parfaitement ; le gradient de D par rapport aux images devient arbitraire et G se fige sur une forme [H, cohérent avec Mescheder §2 et R3GAN, qui montre que RpGAN non régularisé « tourne en rond »].
- **γ = 0,1** : même scénario un peu plus lent (écart ≈ 6,7) ; 3/3 divergent avant l'époque 10 [V]. Ici G **reçoit encore un gradient** (softplus′(6,7) ≈ 1) : ce n'est pas une saturation numérique, c'est un D trop raide dont le gradient n'est pas informatif [H].
- **γ = 10** : stable mais FID 17,7 contre 14,0 et precision 0,67 ; FID encore en baisse à l'époque 100 (18,9 → 17,3 pour s0) ⇒ en partie **plus lent**, pas forcément pire à long terme [V pour la pente, H pour l'asymptote].
- **R2** : aucun gain mesurable (14,4 ± 0,9 contre 14,0 ± 0,9 ; brut 30,5 contre 27,2), coût +5 % [V].
- **Betas d'Adam du papier (0 ; 0,99)** : effondrement (1 seed, 50 ép.) [V]. Probablement lié à notre lr constant et sans calendrier β₂ 0,9 → 0,99 [H].
- **EMA sur la BCE** : −10 % seulement, et ne sauve pas une seed effondrée (impossible par construction : G_ema ne rétroagit pas) [V].
- **Images individuelles** : pas plus nettes (precision CNN plus basse) [V] — voir D6.
- **DRS sur RpGAN + R1 + EMA** : n'aide pas [V] — voir D7.

### 4.3 Dans quel cas

- Le gain de la perte est grand **parce que la baseline est fragile** (BCE sur sigmoïde, sans régularisation, D qui gagne). Sur une baseline déjà régularisée (comme StyleGAN2 dans R3GAN), il serait bien plus petit [L].
- L'EMA aide **quand l'entraînement oscille près d'un bon équilibre** (RpGAN + R1), pas quand l'erreur est une dérive lente (BCE) [V pour l'effet, H pour l'explication].
- L'architecture fixe limite la netteté ; R3GAN améliore surtout par l'architecture, qu'on n'a pas le droit de changer côté G [L].

---

## 5. Corrections des slides et de `training-r3gan.md`

### 5.1 Slides présentées : `masteriasd-datalab-project-a2-group-3/slides/comparaison.tex` (branche `main`)

Numéro = numéro affiché en bas de slide (la page de titre n'est pas comptée, `comparaison.tex:39`).

| Slide (ligne) | Texte actuel | Problème | Proposition |
|---|---|---|---|
| 1 « Three levers » (l. 48) | `Relativistic loss (RpGAN + R1) & Huang et al. 2024` | RpGAN vient de Jolicoeur-Martineau 2019, R1 de Mescheder et al. 2018 ; R3GAN = RpGAN + R1 + **R2**, qu'on n'utilise pas ; l'EMA (moitié du FID final) n'apparaît pas. | `Relativistic loss + R1 + EMA` & `Jolicoeur-Martineau 2019; Mescheder et al. 2018; recipe of Huang et al. 2024 (without R2)` |
| 2 « Protocol » (l. 62-63) | Training : 60k, 100 epochs ; Common : 50k, 200 epochs | Correct. Manque : l'EMA n'est appliquée qu'aux méthodes RpGAN au protocole commun. | Ajouter en note : « Common: EMA only for RpGAN rows ». |
| 10 « Training: the paper » (l. 133-136) | première page de R3GAN | Ce papier rapporte que **RpGAN + R1 (sans R2) échoue** sur StackedMNIST. | À l'oral / dans le rapport : « The paper needs R1 + R2 on StackedMNIST; on our MLP, R1 alone suffices and R2 adds nothing (14.4 vs 14.0). » |
| 11 « Training: method » (l. 141-142) | `L_D = softplus(D(fake) − D(real)) + γ/2 ‖∇D(real)‖²`, `L_G = softplus(D(real) − D(fake))` | **Signes justes, softplus juste, R1 sur les réels seulement, γ/2 juste** (= `utils.py:54-56,73` de `noe/training`). Imprécisions : D = score **avant** sigmoïde ; gradient par rapport à l'image ; espérance sur des paires. | `\nabla_x D(x_{\text{real}})` et une légende : « D: raw score (before the sigmoid). Pairs (real, fake) drawn at random in the batch. » |
| 11 (l. 143) | `Settings: γ = 1, EMA 0.999, 100 epochs` | Le modèle rendu est entraîné **200 époques sur 50k** (protocole commun). | `γ = 1, EMA 0.999 (half-life ≈ 1.2 epoch), Adam 2e-4 (0.9, 0.999), batch 100; 100 epochs (study), 200 (final model)` |
| 12 (l. 151-153) | `0.1: diverges / 1: FID 14.0 / 10: FID 17.7` | Juste ✔. | Option : « 0.1: diverges by epoch 10 (3/3) ». |
| 12 (l. 158-159) | `BCE: 1 run collapses / RpGAN: none` | « RpGAN : none » est faux pour γ = 0 et γ = 0,1 (6/6 divergent) ; et « 1 run » = 1 sur 3. | `BCE: 1 of 3 collapses` / `RpGAN + R1 (γ ≥ 1, with or without R2): 0 of 9` |
| 12 (l. 161) | `Spread between seeds: ± 0.9` | Juste (écart-type 0,88 pour γ = 1 et γ = 10). | `standard deviation over seeds: 0.9` |
| 13 (l. 167) | `Right proportions, no collapsed seed. Single images are not sharper` | Juste ✔ (comparaison à EMA égale). On peut ajouter que les proportions viennent de la perte et pas de l'EMA (sd 1,6 brut contre 3,3). | — |
| 14 « Training: results » (l. 175) | `60k images, 100 epochs, EMA for all, against MNIST test, 3 seeds.` | **Juste pour ce slide** (`configs/sweep/ema100_baseline_s*.yaml` ont `ema: decay: 0.999`). Mais BCE = moyenne de **2** seeds (la figure le dit). | Garder ; ajouter « BCE: 2 stable seeds ». |
| 19 « Common protocol: results » (l. 207-208) | `The loss improves both FIDs. DRS only helps the baseline.` / `Same split, 200 epochs, ...` | **La baseline n'a pas d'EMA** (`configs/compare_local/baseline.yaml`), RpGAN en a une : le gain mélange perte + R1 + EMA. | `Loss + R1 + EMA improve both FIDs` et en source `Baseline without EMA.` ; idéalement ajouter la ligne « RpGAN + R1, no EMA » : CNN 19,7 ± 4,2 (mesuré ici) et son FID Inception (§6, P1a ; seed 1 à 2000 images : égal à la baseline). |
| 21 « Conclusion » (l. 219) | `The loss gives the largest gain.` | Vrai en FID CNN même sans EMA (182 → 19,7). **Probablement faux en FID Inception** : seed 1, 2000 images, baseline 46,1 contre RpGAN + R1 brut 47,3 et + EMA 30,2 (§2.5). | Après P1a : `Loss + R1: right digit proportions (CNN FID 182 → 20). EMA: better images (Inception FID 38 → 22).` En attendant : `RpGAN + R1 + EMA gives the largest gain.` |
| 21 (l. 226, 233) | `RpGAN + R1 + EMA 22.1 / 9.9` ; `Table: common protocol, against MNIST test, 3 seeds.` | Chiffres justes ✔ ; préciser que la baseline n'a pas d'EMA et que le modèle rendu (seed 1) fait 21,2 / 7,6 (meilleur des 3 sur val). | `Baseline without EMA. Final model = seed 1 (best on validation).` |

### 5.2 Rapport de Noé : `wt-comparaison/training-r3gan.md` (identique sur `noe/training`)

| Ligne | Texte | Correction |
|---|---|---|
| 3 | « le FID passe de **58,7** à 14,0 ± 0,9 (3 seeds, …) » | 58,7 ± 4,8 est la moyenne de **2** seeds (la 3ᵉ s'effondre). Écrire « 58,7 (2 seeds stables sur 3) ». |
| 20 | « Avec les deux [R1 et R2], le papier démontre une convergence locale » | R3GAN, Prop. II : RpGAN avec **R1 ou R2** est localement convergent ; ils utilisent les deux pour des raisons empiriques (RpGAN + R1 seul échoue sur StackedMNIST). |
| 22 | « Stacked MNIST … où ils les couvrent tous » | Juste ; ajouter que **RpGAN + R1 seul y échoue** (leur Table 3), contrairement à nous. |
| 41 | EMA : G.pth = G moyenné | Juste ; ajouter : `D.pth` est le D entraîné contre le **G brut** (important pour le DRS). |
| 101 | « son score brut explose et D(x) = D(G(z)) = 1,00 » | Avec RpGAN, le niveau absolu du score est libre : D(x) = D(G(z)) = 1,00 (ou 0,00 aux époques 2-5, `logs/train_rpgan.log`) ne veut rien dire. La preuve que D gagne : `loss_D = 0,000` et `loss_G = 28 à 55` = écart de logits entre vrai et faux. |
| 112 | EMA « appliquée à toutes les configurations, baseline comprise, pour que la comparaison reste juste » | Vrai pour la campagne 100 ép. ; **faux pour le protocole commun** (baseline sans EMA). À signaler dans la partie commune. |
| 132 | « EMA et entraînement plus long aident tout le monde : la baseline passe de 188 à 59 … » | Mélange deux effets. Mesuré ici : à 100 ép., l'EMA fait −10 % sur la BCE et −51 % sur RpGAN + R1. Le 188 → 59 de la baseline vient surtout de la durée. |
| 145 | « pousse le générateur à reproduire la répartition … C'est l'effet anti-oubli de modes décrit par Huang et al. » | La théorie (Sun et al. 2020) parle d'absence de mauvais minima, pas de proportions : présenter comme une interprétation. Vérifié ici : les proportions viennent bien de la perte (sd 1,6 brut contre 3,3 BCE), pas de l'EMA. Mais RpGAN et R1 ne sont pas séparés (BCE + R1 jamais testé). |
| 147 | « C'est précisément ce que le rejet par discriminateur peut corriger » | **Réfuté** au protocole commun : DRS 9,9 → 14,2 (CNN), 22,1 → 22,0 (Inception). |
| 151 | « Le papier justifie R2 par une garantie théorique de convergence » | Voir l. 20 : la garantie locale existe avec R1 seul ; R2 est justifié empiriquement dans le papier. |
| 153 | « FID supérieur à 2000 à l'époque 27 » | Mesuré aux époques 10 et 20 (2091-2672) ; runs arrêtés à l'époque 27-28. |
| 154 | γ = 10 « D devient trop mou » | Plausible ; le FID baisse encore à l'époque 100 : « plus lent » est aussi possible (Mescheder : valeur propre lente −0,025). |
| 160 | « Deux baselines différentes coexistent » | Obsolète : la référence finale est la baseline commune (`baseline.md`). |
| 162 | « le meilleur générateur est `runs/ema100_rpgan_r1_s1/G.pth` … pas encore copié » | **Obsolète** : le modèle rendu est `runs/cmploc_rpgan_r1_ema_s1/G.pth` (200 ép., 50k), copié dans `checkpoints/G.pth` sur `main` (commit fbdd93f). |

### 5.3 Problème de reproductibilité [V]

`train.py` de `main` n'a **ni RpGAN ni EMA** (`git show origin/main:train.py` : 0 occurrence de `ema_update` ; `utils.py` sans `D_train_rpgan`) ; `quentin/comparaison` a RpGAN mais pas l'EMA. **Seule `noe/training` permet de réentraîner le modèle rendu.** À fusionner dans `main` avant le rendu.

---

## 6. Expériences à relancer (Juliet)

Toutes les commandes : depuis la racine du dépôt sur Juliet, branche **`noe/training`** (la seule avec l'EMA), `venv` activé par `scripts/run.sh`. Durées A100 estimées à partir de `baseline.md:189` (14 min pour 200 époques BCE) : RpGAN + R1 ≈ +5 à +50 % (double rétropropagation ; +5 % mesuré sur Mac), mesures Inception + VGG toutes les 10 époques ≈ 20 × 20 s ≈ 7 min.

### P1a — FID Inception + VGG des G bruts du protocole commun (aucun entraînement, ≈ 10 min) — **priorité 1, à faire avant d'écrire la conclusion**

Donne la ligne « RpGAN + R1 sans EMA » du tableau commun dans les deux mesures (la mesure partielle §2.5 suggère qu'en Inception elle vaut la baseline), et teste le DRS avec un D **assorti** à son G (le G brut). Les `G_raw.pth` sont versionnés sur `noe/training`.

```bash
cat > outputs/ema_ablation_manifest.csv <<'EOF'
RpGAN + R1 (G brut),runs/cmploc_rpgan_r1_ema_s0/G_raw.pth
RpGAN + R1 (G brut) + DRS,runs/cmploc_rpgan_r1_ema_s0/G_raw.pth,runs/cmploc_rpgan_r1_ema_s0/D.pth
RpGAN + R1 (G brut),runs/cmploc_rpgan_r1_ema_s1/G_raw.pth
RpGAN + R1 (G brut) + DRS,runs/cmploc_rpgan_r1_ema_s1/G_raw.pth,runs/cmploc_rpgan_r1_ema_s1/D.pth
RpGAN + R1 (G brut),runs/cmploc_rpgan_r1_ema_s2/G_raw.pth
RpGAN + R1 (G brut) + DRS,runs/cmploc_rpgan_r1_ema_s2/G_raw.pth,runs/cmploc_rpgan_r1_ema_s2/D.pth
EOF
bash scripts/run.sh "PYTHONPATH=. python scripts/measure_runs.py outputs/ema_ablation_manifest.csv results/ema_ablation_test.csv" 00:30:00
```

### P1b — Baseline + EMA au protocole commun (3 seeds, ≈ 25 min par run, en parallèle) — **priorité 1**

Créer `configs/compare/baseline_ema.yaml` = copie de `configs/compare/baseline.yaml` avec `name: cmp_baseline_ema`, `save_every: 200` et :

```yaml
ema:
  decay: 0.999
```

```bash
for s in 0 1 2; do bash scripts/run.sh "python train.py --config configs/compare/baseline_ema.yaml --seed $s" 01:00:00; done
# puis mesure identique au tableau commun (dernière époque, G moyenné ET G brut = baseline sans EMA sur la même machine) :
for s in 0 1 2; do
  echo "Baseline + EMA,runs/cmp_baseline_ema_s$s/G.pth"
  echo "Baseline (brut),runs/cmp_baseline_ema_s$s/G_raw.pth"
done > outputs/baseline_ema_manifest.csv
bash scripts/run.sh "PYTHONPATH=. python scripts/measure_runs.py outputs/baseline_ema_manifest.csv results/baseline_ema_test.csv" 00:30:00
```

Avec P1a, on obtient le carré complet {BCE, RpGAN + R1} × {EMA, sans EMA} au protocole commun, en FID Inception et CNN.

### P2 — BCE sur logits ± R1 (sépare « relativité » et « R1 », ≈ 30 min par run, 2 configs × 3 seeds) — **priorité 2**

Le code actuel ignore `r1_gamma` avec `bce` (`utils.py:79-83`). Il faut une branche `ns` (GAN non saturant sur logits, la « GAN + R1 » de R3GAN), ≈ 10 lignes dans `utils.py`, sur le modèle de `D_train_rpgan` :

```python
# D : softplus(-D(réel)) + softplus(D(faux)) + gamma/2 R1 ; G : softplus(-D(faux)). D(.) = D_logits.
D_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()   # au lieu de softplus(fake - real)
G_loss = F.softplus(-fake_logits).mean()                                      # au lieu de softplus(real - fake)
```

Configs `configs/compare/ns_r1_ema.yaml` (`loss: {name: ns, r1_gamma: 1.0, r2_gamma: 0.0}`, `ema: {decay: 0.999}`) et `configs/compare/ns_ema.yaml` (`r1_gamma: 0.0`) ; lancement comme P1b. Le G brut donne la version sans EMA gratuitement. **Ce que ça teste** : (i) `ns` sans R1 contre BCE de la baseline = effet de la saturation numérique de la sigmoïde (2.3) ; (ii) `ns + R1` contre RpGAN + R1 = effet propre de la relativité.

### P3 — compléments (priorité 3)

| Expérience | Commande / config | Coût A100 | Question |
|---|---|---|---|
| RpGAN γ = 0, 3 seeds, 20 époques | `python train.py --config configs/train_rpgan.yaml --seed $s` (avec `epochs: 20`) | ≈ 3 min × 3 | La divergence sans R1 est-elle systématique ? (1 seed aujourd'hui) |
| R3GAN (R1 + R2) + EMA au commun | `configs/compare/r3gan.yaml` + `ema: {decay: 0.999}` | 25 min × 3 | R2 inutile aussi à 200 époques ? |
| γ ∈ {0,3 ; 0,5 ; 2 ; 3} | copies de `configs/sweep/ema100_rpgan_r1_s*.yaml` | 15 min × 12 | Où est l'optimum, et la frontière de divergence ? |
| EMA β = 0,9999 | `ema: {decay: 0.9999}` | 25 min × 3 | Yazıcı : 0,9999 > 0,999 ; demi-vie plus proche de R3GAN (0,5 M images ≈ β 0,99986 à batch 100) |
| 400 époques RpGAN + R1 + EMA | `train.epochs: 400` | 50 min × 3 | Le FID baissait encore à 200 (§2.3) |

---

## 7. Approfondissements possibles

- **R1 + R2 ou R2 seul** : R3GAN Prop. II garantit la convergence locale avec R2 seul ; ce cas n'a jamais été testé.
- **γ adaptatif / décroissant** : R3GAN fait γ 1 → 0,1 (cosinus) sur StackedMNIST (Table 19). Nécessite quelques lignes dans `train.py` (γ par époque, comme `noise_sigma`).
- **Calendrier d'Adam** : R3GAN β₂ 0,9 → 0,99, β₁ = 0, lr 2e-4 → 5e-5. Notre essai β = (0 ; 0,99) d'emblée s'effondre : tester β₁ = 0 avec β₂ = 0,9, ou un lr plus petit.
- **EMA avec montée progressive** (demi-vie 0 → valeur finale, comme R3GAN) et choix de β par validation.
- **DRS cohérent** : réentraîner quelques époques un D **BCE** contre **G_EMA figé**, puis faire le DRS (idée de `training-r3gan.md:161`). P1a teste déjà le cas « D assorti au G brut ».
- **Durée** : choisir l'époque sur le FID de **val** pour toutes les méthodes (y compris la baseline, qui serait meilleure vers l'époque 110 en FID CNN).
- **Mesure des modes** : nombre de chiffres couverts au-delà d'un seuil + KL des proportions (comme le KL de StackedMNIST dans R3GAN), plus parlant que le FID CNN pour l'effet « oubli de modes ».

---

## 8. Sources

**Code et données (lecture seule)**
- `origin/noe/training` (edf14fb) : `train.py:71-83, 126-186` (EMA), `utils.py:25-63` (D_logits, R1/R2, RpGAN), `utils.py:79-121` (BCE), `utils.py:207-210` (`ema_update`), `model.py`, `configs/sweep/*.yaml`, `configs/compare/*.yaml`, `configs/compare_local/*.yaml`, `scripts/compare_local.py`, `scripts/measure_runs.py`, `scripts/sweep_training.py`, `logs/ema100_*.log`, `logs/cmploc_*.log`, `logs/train_*.log`, `logs/sweep_training.log`, `logs/compare_local.log`, `results.csv`, `results/comparison_local.csv`, `runs/*/G.pth`, `runs/*/G_raw.pth`.
- `origin/main` (fbdd93f) : `checkpoints/G.pth`, `checkpoints/D.pth` ; `slides/comparaison.tex` (présentée).
- `wt-comparaison` (7474210) : `training-r3gan.md`, `baseline.md:189`, `slides/make_comparison_figures.py:217-276`.

**Scripts et sorties de cette analyse** (dossier `C:\Users\quent\AppData\Local\Temp\claude\4fbcbedeecc7671c\0451ad2b-5a20-471f-804f-0c5cc895f246\scratchpad\scripts\training\`)
- `log_stats.py` → `log_stats.out` : moyennes ± écart-type, courbes, saturation, protocole commun, proportions.
- `ema_ablation.py` → `ema_ablation.csv` ; `summarize_ablation.py` → `summarize_ablation.out` : G_EMA contre G brut (CNN, test).
- `ema_inception.py` → `ema_inception.csv` : FID Inception (2000 images) G_EMA contre G brut (protocole commun).

**Articles**
- Huang, Gokaslan, Kuleshov, Tompkin. *The GAN is dead; long live the GAN! A Modern GAN Baseline.* NeurIPS 2024, arXiv 2501.05441 (lu en ligne, version HTML) : §2.1-2.4, Table 3 (StackedMNIST), note 2, Prop. II, Table 4 (roadmap), Table 19 (hyperparamètres).
- Jolicoeur-Martineau. *The relativistic discriminator: a key element missing from standard GAN.* ICLR 2019 (local `papiers/training/`) : Tables 2-3.
- Mescheder, Geiger, Nowozin. *Which Training Methods for GANs do actually Converge?* ICML 2018 (local) : Lemme 3.3, §4.1-4.2, Théorème 4.1, annexe (γ = 10).
- Yazıcı, Foo, Winkler, Yap, Piliouras, Chandrasekhar. *The Unusual Effectiveness of Averaging in GAN Training.* ICLR 2019 (arXiv 1806.04498) : Table 2, Table 4.
- Sun, Fang, Schwing. *Towards a Better Global Loss Landscape of GANs.* NeurIPS 2020 (cité via R3GAN, non relu).
