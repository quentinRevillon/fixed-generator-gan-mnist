# Volet Inférence : le DRS fait-il baisser les performances ?

> **Réponse courte à la question du professeur.** Sa phrase n'est vraie qu'en partie.
> - **Baseline :** le DRS améliore le FID Inception et la précision sur toutes les sources et à tous les percentiles (seules exceptions : KeepTraining au percentile 99, et le FID CNN au percentile 50 sur test, +1 à +4, dans le bruit). Le rappel VGG baisse très peu (−0,014).
> - **RpGAN + R1 + EMA, FID Inception (la mesure du classement) :** effet **nul** (22,14 → 22,04 ; écarts par seed −0,35 / +0,71 / −0,66). Précision et rappel ne bougent pas.
> - **RpGAN + R1 + EMA, FID du petit CNN :** dégradation **réelle et reproductible** (9,9 → 14,2 ; pire sur les 3 seeds, de +2,9 à +5,6). C'est ce chiffre que montre la slide de conclusion.
> - **Pourquoi (vérifié par nos diagnostics) :** contre G_EMA, le D de RpGAN ne sait presque plus distinguer les vraies images des fausses (AUC 0,55-0,56 contre 0,75-0,76 pour la baseline). Le DRS tire donc presque au hasard (43 % gardées au lieu de 30 %), et le peu de préférence qui reste ne va pas dans le bon sens : D garde plus de « 6 » et moins de « 8 », alors qu'il faudrait l'inverse. Le mélange des chiffres se déforme (c'est toute la hausse pour la seed 0, la moitié pour la seed 1) ; le petit CNN, qui est un classifieur de chiffres, le voit ; Inception presque pas.
> - **Ce qui ne l'explique pas :** la constante arbitraire de la perte relativiste (le DRS y est insensible), la perte de modes (le rappel ne bouge pas), le seul décalage EMA (le DRS échoue aussi sur le G brut, avec un D assorti).
> - **Ce qui répare en partie :** recalibrer D comme Azadi le fait pour SAGAN (une couche logistique réentraînée sur réels contre G figé). Sur la baseline, le gain en FID CNN devient énorme (174 → 111) ; sur RpGAN, le DRS redevient neutre (un peu mieux ou un peu moins bien que sans tri selon la seed). À confirmer en Inception sur GPU.

Notions utilisées :
- **D, d(x)** : le discriminateur et sa note avant la sigmoïde (le *logit*). **G_EMA** : la moyenne mobile des poids de G, celle qu'on échantillonne ; **G brut** : les poids de G à la dernière étape.
- **Rapport de densités** r(x) = p_data(x) / p_G(x) : combien de fois une image est plus fréquente dans les vraies données que chez G. Le DRS veut garder chaque image avec une chance proportionnelle à r(x).
- **AUC** : probabilité que D note une vraie image plus haut qu'une fausse, tirées au hasard. 0,5 = D ne distingue rien ; 1 = D distingue tout.
- **Recalibrer D** : réajuster sa sortie sur des données neuves (réelles contre G figé) pour qu'elle redevienne une estimation de log r(x).
- **TV (distance de variation totale) entre mélanges de chiffres** : part des images qui devraient changer de chiffre pour retrouver les proportions de MNIST test.

---

## 1. Attendu (littérature)

### 1.1 Les hypothèses du DRS (Azadi et al. 2019, section 3.1)
La formule « garder x avec la chance e^{d(x) − d_M} » n'est exacte que si :
1. **D est entraîné avec une sigmoïde et la BCE** (entropie croisée binaire) ;
2. **D est optimal** pour le G final : D*(x) = p_data / (p_data + p_G), donc d(x) = log r(x) (équations 3 à 5) ;
3. M = max r(x) est connu (estimé sur 10 000 images, puis mis à jour) ;
4. supports communs. Les auteurs listent eux-mêmes ces limites (section 3.2, points 1 à 5).

Ajouts pratiques : **KeepTraining** (continuer d'entraîner D seul, G figé, avec arrêt sur la validation) et **γ** (décalage qui augmente le taux d'acceptation ; percentile 95 sur le jeu 2D, 80 sur ImageNet). Avec γ, le DRS **ne vise plus exactement p_data**, même avec un D parfait (Turner et al. 2019, section 2 et figure 2).

Point clé pour nous : SAGAN est entraîné avec la perte *hinge*, pas la BCE. Azadi **ajoute donc une couche entièrement connectée au-dessus de D, entraînée en BCE sur 10 000 images générées et 10 000 vraies** (section 4.2). Autrement dit, quand la perte n'est pas la BCE, le papier ne prend pas D tel quel : il le recalibre.

### 1.2 Les gains annoncés
| Source | Modèle, données | Sans DRS | Avec DRS |
|---|---|---|---|
| Azadi, tableau 1 | GAN MLP, 25 gaussiennes 2D, γ au 95e percentile | 70 ± 9 % d'échantillons « de qualité », 24,8 modes | 90 ± 2 %, 24,8 modes |
| Azadi, tableau 2 | SAGAN, ImageNet 128, γ au 80e percentile | FID 18,21, IS 52,34 | FID 17,14, IS 61,44 |
| Azadi, tableau 2 | SAGAN amélioré | FID 14,79, IS 62,36 | FID 13,57, IS 76,08 |
| Ansari 2021, tableau 3 (repris de Turner) | DCGAN, CIFAR-10, D calibré | IS 2,88 | DRS 3,07 ; MH-GAN 3,38 |
| Verine 2024, tableau 3 (OBRS) | diffusion EDM, CIFAR-10, classifieur dédié | FID 1,97, P 77,9, R 86,6 | FID 1,57 à 25 % d'acceptation, P 78,5, R 86,7 |

Ordre de grandeur : **−6 à −8 % de FID** sur des modèles forts, rappel inchangé. Azadi (annexe A, figure 7) note aussi que le gain sature quand on rejette davantage.

### 1.3 Ce que disent les autres papiers sur les conditions de réussite
- **Turner et al. 2019 (MH-GAN), section 4 :** « les probabilités de D doivent non seulement donner une bonne AUC, mais aussi être bien calibrées ». Ils recalibrent D sur un jeu tenu à part (régression logistique, isotonique ou bêta) ; « la calibration aide dans tous les cas ». Ils observent aussi que le taux d'acceptation dépend surtout du recouvrement des notes de D entre vrais et faux, donc de l'AUC.
- **Verine et al. 2024 (OBRS), section 5.2 :** un D entraîné avec la hinge « n'est pas adapté à l'estimation de densité » ; ils affinent donc D avant de rejeter. L'acceptation optimale à budget K est min(1, r(x)·c_K / M) ; elle augmente la précision et garde le rappel (théorème 3.3), **à condition que r soit bien estimé**.
- **Issenhuth et al. 2021, figure 2 :** en grande dimension, quand les variétés réelle et générée se recouvrent mal, le DRS ne garde que leur intersection.
- **Che et al. 2020 (DDLS)** et **Ansari et al. 2021 (DGflow)** utilisent aussi D comme estimateur de log r, toujours avec un D calibré (« cal » dans leurs tableaux).

**Attendu pour nous :** le DRS doit aider tant que D (1) sait encore distinguer les vraies images de celles de G et (2) donne une note proportionnelle à log r. Un D entraîné avec une autre perte que la BCE doit être recalibré.

---

## 2. Observé chez nous

### 2.1 Tableau de vérification, toutes sources
Signe : « mieux » = le DRS améliore la mesure (FID plus bas, précision ou rappel plus haut).

| Source | G | Mesure, référence | Sans DRS | DRS p80 | Verdict |
|---|---|---|---|---|---|
| `results/results_prof.csv:38-46` | baseline ép. 200 (`baseline_200ep_prof`), 3 tirages | FID Inception, val | 36,69 | 31,59 | **mieux** (p50 32,77 ; p95 31,33 ; p99 31,79 : tous mieux) |
| idem | idem | précision / rappel VGG | 0,490 / 0,459 | 0,532 / 0,456 | précision mieux, rappel égal |
| `results/results_prof.csv:53-64` | idem, D après KeepTraining | FID Inception, val | 36,69 | 36,06 | à peine mieux ; p95 36,82 ≈ égal ; **p99 38,37 : pire** |
| `results/results_prof.csv:11-37` | baseline ép. 120 (`baseline_200ep`) | FID Inception, val | 56,34 | 52,18 | mieux (KeepTraining aussi, mais moins) |
| `results/results.csv` (lignes `baseline_200ep_prof`) | baseline ép. 200, 3 tirages | FID CNN, val | 168,9 | 145,7 | mieux (p50 164,7 ; p95 121,1 ; p99 104,1) |
| `results/results.csv` (expérience 1, `inference-drs.md:221-229`) | ancien `checkpoints/G.pth` | FID CNN, val | 97,6 | DRS réglé par K et OBRS, K = 1,5 à 50 | mieux partout |
| `results/comparison_local.csv` (Noé, protocole commun) | baseline, 3 seeds d'entraînement × 1 tirage | FID Inception, test | 38,66 ± 1,23 | 34,36 ± 0,73 | **mieux, 3/3 seeds** (−6,0 / −2,2 / −4,6) |
| idem | idem | précision / rappel VGG | 0,466 / 0,508 | 0,507 / 0,494 | précision mieux 3/3 ; **rappel −0,014, pire 3/3** (faible) |
| idem | idem | FID CNN, test | 181,8 ± 19,4 | 154,0 ± 30,8 | mieux 3/3 (−4 / −19 / −61, très bruité) |
| idem | RpGAN + R1 + EMA | FID Inception, test | 22,14 ± 1,26 | 22,04 ± 1,10 | **neutre** (−0,35 / +0,71 / −0,66) |
| idem | idem | précision / rappel VGG | 0,589 / 0,587 | 0,586 / 0,581 | neutre |
| idem | idem | **FID CNN, test** | **9,93 ± 2,47** | **14,17 ± 2,28** | **pire, 3/3 seeds** (+4,3 / +5,6 / +2,9) |
| idem | idem | précision / rappel CNN | 0,738 / 0,828 | 0,743 / 0,825 | neutre |
| idem | RpGAN + R1 + EMA + bruit 0,3 | FID Inception, test | 21,44 ± 0,58 | 21,93 ± 1,15 | neutre à légèrement pire (+0,37 / +1,62 / −0,54) |
| idem | idem | FID CNN, test | 9,01 ± 1,04 | 13,90 ± 5,82 | **pire, 3/3 seeds** (+12,0 / +2,1 / +0,6) |
| nos diagnostics (§6.1) | RpGAN seed 1, 2 tirages ; seed 0, 1 tirage | FID CNN, test | 7,60 / 7,72 ; 12,52 | 13,19 / 11,89 ; 16,77 | pire ; p50 et p95 aussi |
| nos diagnostics (§6.1) | baseline ép. 200 ; baseline commune seed 1 | FID CNN, test | 174,0 ; 201,0 | 158,9 ; 182,5 | mieux (p50 : légèrement pire, 177,7 et 202,0) |

Moyennes et écarts par seed : calculés depuis `results/comparison_local.csv` (identique sur `origin/noe/training` et `quentin/comparaison`, vérifié par `diff`). Les ± sont des écarts-types sur 3 seeds d'entraînement ; Noé ne fait **qu'un tirage** par seed (`scripts/measure_runs.py`, `SEEDS = (0,)`).

### 2.2 Ce que la phrase du prof a de vrai
- **Fausse pour la baseline**, sur toutes les sources, sauf une baisse de rappel VGG très faible (−0,014) mais systématique sur le protocole commun, et le FID CNN au percentile 50 contre test (+3,7 et +1,0 dans nos diagnostics, un seul tirage, alors que le bruit entre tirages vaut ±5 pour la baseline, `results.csv`).
- **Fausse en FID Inception pour RpGAN** : 22,1 → 22,0, à l'intérieur du bruit entre seeds.
- **Vraie pour RpGAN en FID CNN** : +4,2 en moyenne, pire sur 3 seeds sur 3, et le même résultat avec le bruit d'instance. Nos diagnostics le reproduisent au bit près (FID CNN 7,597 → 13,193 pour la seed 1, tirage 0, identique à `comparison_local.csv`).

Donc : **« le DRS n'aide plus quand le modèle est bon, et il dégrade le FID du petit CNN »** est la formulation exacte.

---

## 3. Divergences et pourquoi

### 3.1 Vérifications dans le code
- **Perte RpGAN** (`utils.py` de `origin/noe/training`, `D_train_rpgan`) : `softplus(fake_logits − real_logits) + r1_gamma/2 · ||∇D(réel)||²`. Seules les **différences** de notes comptent : ajouter une constante à d ne change rien à la perte. Rien n'oblige donc d à valoir log r. **Vérifié.**
- **EMA** (`train.py`, `ema_update`, lignes 79-80 et 183-186) : seule G a une moyenne mobile. `G.pth` = G_EMA, `G_raw.pth` = G brut, `D.pth` = D entraîné **contre G brut**. **Vérifié.**
- **Checkpoint du dépôt** : `checkpoints/G.pth` et `checkpoints/D.pth` du dépôt principal ont le même MD5 que `runs/cmploc_rpgan_r1_ema_s1/G.pth` et `D.pth` (commit 0a03c96). C'est donc **G_EMA** de RpGAN + R1 + EMA, seed 1, dernière époque, avec son D. **Vérifié.**
- **Le DRS est invariant à une constante ajoutée à d** : F̂ ne dépend que de d − d_M, et d_M bouge avec d. Le « défini à une constante près » de RpGAN **n'est donc pas un problème en soi**. Ce qui compte, c'est l'**échelle** de d (pente) et surtout **sa capacité à classer** les images selon r(x).
- **Avec γ au percentile, le DRS revient à** a(x) ≈ σ(d(x) − q₈₀), où q₈₀ est le 80e percentile des notes du batch (pour d loin de d_M, F̂ ≈ d − d_M). Si toutes les notes sont proches, chaque image a ≈ 50 % de chances : **le DRS devient un tirage au hasard**.

### 3.2 Diagnostics (voir §6.1 pour le détail)
Notes de D sur 10 000 images de MNIST test et 10 000 images de G (`scratchpad/results/drs/scores_*.csv`) :

| Modèle | d réels (moy. ± é.-t.) | d sur G (moy. ± é.-t.) | AUC de D | AUC de D recalibré (fc3) | log E_G[e^d] (0 si d = log r) | pente de Platt a (1 si bien calibré) |
|---|---|---|---|---|---|---|
| baseline ép. 200 (`bl200_prof`) | 1,06 ± 2,43 | −0,93 ± 1,55 | **0,756** | 0,831 | 0,50 | 0,62 |
| baseline commune seed 1 (`bl_s1`) | 0,12 ± 2,66 | −1,83 ± 1,46 | **0,752** | 0,852 | −0,63 | 0,62 |
| RpGAN + R1 + EMA seed 1, G_EMA (`rp_s1_ema` = `checkpoints/`) | 0,92 ± 0,21 | 0,88 ± 0,21 | **0,548** | 0,772 | 0,90 | 0,77 |
| RpGAN + R1 + EMA seed 0, G_EMA (`rp_s0_ema`) | 0,19 ± 0,23 | 0,14 ± 0,23 | **0,563** | 0,786 | 0,17 | 0,98 |
| RpGAN + R1 + EMA seed 1, **G brut** (`rp_s1_raw`) | 0,92 ± 0,21 | 0,80 ± 0,19 | **0,663** | 0,875 | 0,82 | 2,85 |

Lecture : chez RpGAN, les notes des vrais et des faux ont presque la même moyenne et un écart-type dix fois plus petit que chez la baseline. La moyenne elle-même dérive d'un seed à l'autre (0,92 contre 0,19) : c'est la constante arbitraire de la perte relativiste.

### 3.3 Les hypothèses, une à une
**H1 — Avec RpGAN, d n'est pas log r.** *Vérifiée en partie, mais pas pour la raison attendue.*
- La constante arbitraire existe : sur G_EMA, log E_G[e^d] = 0,90 au lieu de 0 (seed 1). Mais elle est sans effet sur le DRS (§3.1) ; la recalibration de Platt (d' = a·d + b, a = 0,77) ne change presque rien : FID CNN 12,8 / 11,3 contre 13,2 / 11,9 sans recalibration.
- Le vrai problème est que **d ne classe plus les images** : AUC 0,548 entre MNIST test et G_EMA (baseline : 0,756). Les notes des vrais et des faux se superposent presque (moyennes 0,92 et 0,88, écart-type 0,21 ; baseline : 1,06 et −0,93, écarts-types 2,4 et 1,5). Fichiers : `results/drs/hist_*.csv`.
- Conséquence directe : 43 % d'images gardées au percentile 80, contre 30 % pour la baseline (aussi dans `comparison_local.csv` : 0,426 contre 0,304). Le DRS est presque un tirage au hasard.
- Ce qui reste de préférence est **sans rapport avec r** : pour la seed 1, acceptation de 37,6 % pour les « 3 » et 38,9 % pour les « 8 », contre 50,2 % pour les « 6 ». Pour corriger le mélange des chiffres, il faudrait garder plus de « 8 » et de « 9 » (sous-représentés : 8,5 % et 8,1 %). Corrélation entre l'acceptation par chiffre observée et celle qui corrigerait le mélange (π_test / π_G) : **−0,17** (seed 1), +0,20 (seed 0), −0,01 (G brut), −0,24 (baseline) (`scratchpad/results/drs/classes_*.csv`). Aucun D ne corrige le mélange des chiffres ; chez la baseline, ce défaut est masqué par un vrai gain de qualité à l'intérieur de chaque chiffre (H3).

**H2 — D a vu G brut, on échantillonne G_EMA.** *Vérifiée en partie : elle aggrave, mais n'est pas la cause première.*
- Même D (seed 1) : AUC 0,663 contre G brut, 0,548 contre G_EMA. L'EMA déplace les images vers une zone où D, qui n'a jamais vu G_EMA, ne distingue presque plus rien.
- Mais le DRS **ne marche pas non plus sur G brut**, où D est « assorti » : FID CNN 15,0 → 17,0 (p80), 16,8 (p50), 17,3 (p95). La perte RpGAN + R1 elle-même ne donne pas un D utilisable comme estimateur de r.
- L'EMA seule vaut bien plus que tout tri : G brut 15,0 → G_EMA 7,6 (FID CNN, seed 1).
- Platt sur G brut donne a = 2,85 (D *pas assez* sûr de lui) ; corriger l'échelle durcit le tri et dégrade encore (21,4).

**H3 — Rejeter fait chuter le rappel quand G est déjà bon.** *Fausse dans sa forme « perte de modes ».* Le rappel ne bouge pas (VGG 0,587 → 0,581 ; CNN 0,828 → 0,825 ; diagnostics 0,831 → 0,835). Ce qui se dégrade :
- **le mélange des chiffres** : TV à MNIST test 0,044 → 0,054 (moyenne des 3 seeds, `comparison_local.csv`) ; 0,043 → 0,057 dans nos diagnostics ;
- **la répartition à l'intérieur de chaque chiffre**, selon la seed. On ramène les images gardées par le DRS au mélange de chiffres de « sans tri » (variante `drs_p80_plain_mix`) :
  - seed 1 : 10,2 / 9,8, contre 13,2 / 11,9 pour le DRS et 7,6 / 7,7 sans tri → **environ la moitié de la hausse vient du mélange, l'autre de l'intérieur des chiffres** ;
  - seed 0 : 11,7, contre 16,8 pour le DRS et 12,5 sans tri → **toute la hausse vient du mélange** ; à mélange égal, le DRS est même un peu meilleur.
- Pour comparaison, la baseline : à mélange égal, le DRS **améliore** l'intérieur des chiffres (`bl200_prof` : 165,4 contre 174,0 sans tri ; `bl_s1` : 183,3 contre 201,0), et c'est presque tout son gain (DRS 158,9 et 182,5).
- Conclusion : la cause robuste de la hausse du FID CNN chez RpGAN est **la déformation du mélange des chiffres** (D a des préférences de chiffres arbitraires) ; s'y ajoute, selon la seed, une légère perte de typicité. Pas de perte de modes.

**H4 — D non recalibré ; γ par percentile.** *Vérifiée pour la recalibration : elle répare en partie.*
- Recalibration « à la Azadi » : on remplace la dernière couche de D par une régression logistique sur les 256 sorties de fc3, entraînée sur 10 000 images de `val` contre 10 000 images de G_EMA (c'est la couche BCE ajoutée par Azadi pour SAGAN). L'AUC sur test passe de 0,55 à 0,77 : **l'information existe dans D, sa dernière couche ne l'exploite pas**.
- RpGAN, FID CNN avec ce D recalibré (DRS p80 / OBRS K = 3), à comparer à « sans tri » et au DRS d'origine :
  - seed 1 : 10,3 / 9,8 (DRS), 10,8 / 9,9 (OBRS) ; sans tri 7,6 / 7,7 ; DRS d'origine 13,2 / 11,9 → **moins mauvais, toujours pire que sans tri** ;
  - seed 0 : 12,0 (DRS), **10,7** (OBRS) ; sans tri 12,5 ; DRS d'origine 16,8 → **légèrement meilleur que sans tri** ;
  - G brut seed 1 : 13,9 / 14,7 ; sans tri 15,0 → légèrement meilleur.
  - Dans tous les cas, la précision CNN monte (0,738 → 0,760-0,770) et le mélange des chiffres se rapproche de MNIST (TV 0,043 → 0,036). **La recalibration répare le sens du tri ; le gain qui reste à prendre sur un G aussi bon est petit, et son signe dépend de la seed.** À trancher en Inception (§6.2, n° 1).
- Baseline : la recalibration donne un **gros** gain en FID CNN. `bl200_prof` : 174,0 sans tri → 158,9 DRS → **110,9** DRS avec D recalibré → **86,9** OBRS (K = 3) avec D recalibré. `bl_s1` (Noé) : 201,0 → 182,5 → **118,5** → **100,3**. Le rappel CNN remonte au lieu de baisser (`bl_s1` : 0,695 avec DRS, 0,732 avec D recalibré) et le mélange des chiffres se rapproche de MNIST (TV 0,258 → 0,192). À confirmer en Inception (§6.2, n° 2).
- Platt seul ne suffit pas : pour la baseline, a = 0,62, donc D est « trop sûr » d'un facteur 1,6 ; corriger seulement l'échelle adoucit le tri et réduit le gain (168,6 au lieu de 158,9).
- γ par percentile n'est pas en cause : à taux d'acceptation voisin, OBRS (qui n'a pas de percentile) fait à peu près pareil avec le D d'origine (RpGAN seed 1 : 10,9 / 11,4 contre 13,2 / 11,9 ; seed 0 : 16,7 contre 16,8).
- KeepTraining (D entier ré-entraîné, `keep_training.py`) dégrade au contraire la baseline (`inference-drs.md:95-106`). *Hypothèse non vérifiée* : D est ré-entraîné sur les images d'entraînement qu'il connaît déjà, et devient très sûr de lui (BCE de validation 1,16 → 0,22) ; les notes s'étalent, le DRS se rapproche d'un seuil dur, que l'ablation d'Azadi (annexe A) montre nuisible aux modes. Le mélange des chiffres se déforme (« 9 » de 13,7 % à 21 %, `results_prof.csv:56-58`). Une régression logistique sur les features, convexe et régularisée, évite ce piège.

**H5 — Pourquoi Inception ne bouge pas (22,1 → 22,0) quand le CNN se dégrade (9,9 → 14,2).** *Vérifiée pour le CNN, hypothèse pour Inception (Inception non recalculé sur CPU).*
- Le petit CNN est un **classifieur de chiffres** : ses features séparent fortement les 10 chiffres. Un petit changement de mélange ou de forme typique d'un chiffre déplace la moyenne et la covariance des features, donc le FID. Le FID CNN de RpGAN est petit (≈ 10) : +3 à +5 points dus au mélange et à la typicité sont donc très visibles en relatif.
- Inception, entraîné sur ImageNet, encode des textures et des formes génériques : les chiffres y sont moins séparés et le FID (≈ 22) est dominé par l'écart de style MLP contre MNIST, que le DRS ne change pas.
- Pour la baseline, c'est l'inverse : le DRS améliore la netteté des images (précision VGG +0,04), ce qu'Inception voit.
- Mesure de la sensibilité au mélange (vérifiée) : les images « sans tri » de la baseline ép. 200, sous-échantillonnées pour avoir exactement le mélange de chiffres de MNIST test, passent de **174,0 à 30,3** en FID CNN. **Environ 145 des 174 points du FID CNN de la baseline viennent du seul mélange des chiffres.** À mélange égal (celui de test), le DRS p80 fait 26,4 contre 30,3 : le vrai gain « qualité » du DRS sur la baseline est d'environ 13 %, comparable aux 14 % du FID Inception (36,7 → 31,6). Pour RpGAN, ce rééquilibrage ne change presque rien (7,6 → 7,1 ; 7,7 → 7,7), car son mélange est déjà proche de test (TV 0,043).
- Conséquence pour les slides : le FID CNN de la baseline (182 → 154) mesure surtout un changement de mélange des chiffres, et ses ± sont énormes (écart-type 19 à 31 entre seeds). Il ne faut pas le comparer à celui de RpGAN en relatif.

### 3.4 Synthèse du mécanisme
1. Le DRS suppose que **d classe les images selon r(x)**. C'est vrai pour la baseline (G encore loin des données, D en BCE qui distingue bien : AUC 0,76).
2. Avec RpGAN + R1 + EMA, G_EMA est presque indiscernable des données **pour ce D** (AUC 0,55) : r(x) ≈ 1 selon D, il n'y a presque rien à corriger, et le DRS tire presque au hasard.
3. Le petit signal restant de D ne mesure pas r (corrélation de −0,17 à +0,20 avec la correction nécessaire du mélange des chiffres, selon le modèle) : il ajoute un biais au lieu d'en retirer un. Inception ne le voit pas ; le CNN, très sensible au mélange des chiffres, le voit.

---

## 4. Ce qui marche / ne marche pas, et dans quel cas
| Cas | DRS | Mécanisme |
|---|---|---|
| G clairement imparfait, D en BCE (baseline) | **marche** : −4,3 FID Inception (test), précision +0,04, rappel −0,01 | D distingue bien (AUC 0,76), d suit à peu près log r |
| idem, D recalibré par une couche logistique sur ses features | **marche beaucoup mieux** en FID CNN (174 → 111 au lieu de 159 ; 201 → 119 au lieu de 183), à confirmer en Inception | AUC 0,76 → 0,83-0,85 ; mélange des chiffres corrigé en partie |
| idem, D entier ré-entraîné (KeepTraining) | moins bien que D tel quel, pire au p99 | D trop sûr, quasi seuil dur, mélange des chiffres déformé (hypothèse) |
| G proche des données, D RpGAN tel quel | **neutre** en Inception, **nuisible** en CNN | D ne distingue plus (AUC 0,55) ; tri quasi aléatoire + biais de chiffres |
| idem, D recalibré | neutre en FID CNN (seed 1 : pire que sans tri ; seed 0 : un peu mieux) | AUC remonte à 0,77-0,79, mais il reste peu à corriger quand G est déjà bon |
| G brut de RpGAN, D assorti | nuisible aussi (15,0 → 17,0) | la perte RpGAN + R1 ne fait pas de D un estimateur de r |
| Percentile plus haut | baseline : FID Inception qui sature après p80 ; RpGAN : pire (seed 1 : p50 11,4 → p80 13,2 → p95 13,7) | plus on rejette, plus le biais de D pèse |

Règle pratique : **mesurer l'AUC de D (réels de validation contre le G qu'on échantillonne) avant d'utiliser le DRS.** Près de 0,5, le DRS n'a rien à corriger.

---

## 5. Corrections des slides (`slides/comparaison.tex`) et de `inference-drs.md`

### 5.1 Slides
1. **« DRS only helps the baseline » (ligne 160).** Exact, mais il manque la moitié de l'information. Proposer : « DRS helps the baseline. On RpGAN it changes nothing in Inception FID (22.1 → 22.0) and worsens the CNN FID (9.9 → 14.2, 3 seeds of 3): D no longer separates real from generated (AUC 0.55). »
2. **Slide « Sampling: method » (lignes 136-142).**
   - « γ ≈ −6.4 » ne vaut que pour la baseline ép. 200 (d_M = 6,67, 80e percentile de d = 0,26 ; `inference-drs.md:145,167`). Pour RpGAN, γ ≈ 0 à 0,2 (seed 1 : d_M ≈ 1,74, 80e percentile de d ≈ 1,06 ; seed 0 : d_M ≈ 0,92 ; `scores_*.csv`). Écrire « γ ≈ −6.4 for the baseline ».
   - « D as trained » : préciser « D as trained, no recalibration (the paper adds a BCE layer for non-BCE losses) ». C'est exactement ce qui manque pour RpGAN.
   - La formule F̂ et ε = 10⁻⁶ sont correctes (`sampling.py:187-190`, éq. 8 du papier). « d_M: best score seen » : préciser « estimated on 10,000 images, then updated ».
   - Ajouter l'hypothèse : « assumes logit(D) = log p_data / p_G (D optimal for BCE) ».
3. **Slide « Sampling: results »** : la légende dit « Generator at epoch 200, against validation, 3 draws » ; ajouter « baseline G, one training run » (les ± ne couvrent que le tirage, `inference-drs.md:60`).
4. **Slide « Compared with the papers », ligne « DRS / better FID, modes kept / same »** : vrai pour la baseline seulement. Proposer : « same on the baseline; none on RpGAN ». Sur le protocole commun, le rappel VGG de la baseline baisse un peu (−0,014, 3/3 seeds) : « modes kept » reste défendable, mais pas « recall up ».
5. **Slide « Protocol »** : préciser le nombre de tirages : « Sampling: one training run, 3 draws ; Common: 3 training seeds, 1 draw each » (`scripts/measure_runs.py`, `SEEDS = (0,)`). La slide « Sampling: results » (val, une seule baseline) et la conclusion (test, 3 seeds) ne sont pas comparables entre elles. Sur « Sampling: results », ajouter que le FID CNN de la baseline est dominé par le mélange des chiffres (174 → 30 si l'on rééquilibre les chiffres, §3.3 H5) : sa forte baisse au p95-p99 traduit surtout un rééquilibrage, pas une meilleure qualité.
6. **Conclusion** : la ligne « + DRS 22.0 / 14.2 » est juste ; ajouter la colonne précision / rappel ou une note « DRS: no change in precision/recall ».

### 5.2 `inference-drs.md`
- Ligne 3 et ligne 6 (« Le tri fonctionne ») : vrai pour la baseline seulement ; ajouter la limite RpGAN (§2.1).
- Ligne 7 (réglage recommandé, « avec le discriminateur d'origine ») : faux pour le modèle final proposé (RpGAN) ; recommander **sans DRS** pour RpGAN, ou DRS avec D recalibré si l'expérience §6.2 le confirme en Inception.
- Ligne 116 (« `checkpoints/` contient encore l'ancien modèle ») : **périmé**. `checkpoints/G.pth` et `D.pth` du dépôt principal sont maintenant RpGAN + R1 + EMA seed 1 (MD5 identiques à `runs/cmploc_rpgan_r1_ema_s1/`).
- Ligne 118 (« Tester le DRS sur les meilleurs modèles de Noé ») : fait (`comparison_local.csv`) ; le dire et donner le résultat.
- Ligne 200 (« Ce FID continue de baisser avec le rejet, car il est très sensible au mélange des chiffres ») : bon constat ; il explique aussi la hausse sur RpGAN (§3.3 H5).
- Annexe B, ligne 145 (« la meilleure note observée (ici 6,67) ») : préciser que c'est pour la baseline.
- Lignes 97-106 (KeepTraining « pas d'explication vérifiée ») : ajouter l'hypothèse §3.3 H4 et le résultat de la recalibration par couche logistique.

---

## 6. Expériences

### 6.1 Faites (CPU local, `torch.set_num_threads(2)`)
Script : `scratchpad/scripts/drs/diag_drs.py` (importe `sampling.py`, `evaluate.py`, `model.py`, `utils.py`, `data.py` du worktree, sans rien y écrire). Sorties : `scratchpad/results/drs/` (`scores_*.csv`, `metrics_cnn_*.csv`, `classes_*.csv`, `hist_*.csv`). Toutes les mesures : petit CNN, 10 000 images, contre MNIST test. Recalibrations ajustées sur `val` (10 000 réelles) contre 10 000 images de G (seed 100), AUC mesurée sur test contre d'autres images de G (seed 200).

Modèles : `bl200_prof` = `wt-comparaison/runs/baseline_200ep_prof/G_best.pth, D_best.pth` ; `bl_s1` = `runs/cmploc_baseline_s1/` (Noé) ; `rp_s1_ema` = `runs/cmploc_rpgan_r1_ema_s1/G.pth` (= `checkpoints/G.pth`) ; `rp_s1_raw` = même run, `G_raw.pth` ; `rp_s0_ema` = `runs/cmploc_rpgan_r1_ema_s0/`.

FID CNN contre MNIST test (plusieurs valeurs = plusieurs tirages, séparées par « / ») ; entre parenthèses : précision et rappel CNN, TV du mélange de chiffres à MNIST test, taux d'acceptation.

| Variante | bl200_prof | bl_s1 | rp_s1_ema | rp_s0_ema | rp_s1_raw |
|---|---|---|---|---|---|
| plain | 174,0 (P 0,836, R 0,755, TV 0,258) | 201,0 (P 0,847, R 0,728, TV 0,261) | **7,6 / 7,7** (P 0,738, R 0,833, TV 0,043) | **12,5** (P 0,733, R 0,811, TV 0,046) | 15,0 (P 0,711, R 0,794, TV 0,059) |
| drs_p50 | 177,7 (50 %) | 202,0 (48 %) | 11,4 / 11,3 (50 %) | 13,4 (51 %) | 16,8 (51 %) |
| **drs_p80** (réglage présenté) | 158,9 (P 0,862, R 0,743, TV 0,231, 30 %) | 182,5 (P 0,880, R 0,695, TV 0,250, 31 %) | **13,2 / 11,9** (P 0,742, R 0,830, TV 0,058, 43 %) | **16,8** (P 0,731, R 0,812, TV 0,055, 42 %) | 17,0 (TV 0,054, 43 %) |
| drs_p95 | 128,6 (14 %) | 151,3 (17 %) | 13,7 / 13,2 (34 %) | 18,1 (32 %) | 17,3 (35 %) |
| drs_p80_platt | 168,6 (35 %) | 189,4 (35 %) | 12,8 / 11,3 (44 %) | 16,7 (42 %) | 21,4 (39 %) |
| **drs_p80_featlr** (D recalibré) | **110,9** (P 0,852, R 0,747, TV 0,192, 32 %) | **118,5** (P 0,860, R 0,732, TV 0,183, 31 %) | 10,3 / 9,8 (P 0,760, R 0,825, TV 0,036, 31 %) | 12,0 (P 0,760, R 0,810, TV 0,043, 31 %) | 13,9 (TV 0,026, 28 %) |
| obrs_K3 | 142,7 (34 %) | 184,3 (33 %) | 10,9 / 11,4 (44 %) | 16,7 (46 %) | 16,1 (47 %) |
| **obrs_K3_featlr** | **86,9** (P 0,853, R 0,754, TV 0,169) | **100,3** (P 0,859, R 0,731, TV 0,171) | 10,8 / 9,9 (P 0,770, R 0,833, TV 0,036) | **10,7** (P 0,768, R 0,806, TV 0,035) | 14,7 |
| drs_p80_plain_mix | 165,4 | 183,3 | 10,2 / 9,8 | 11,7 | 15,7 |
| drs_p80_test_mix | 26,4 | — (pas assez de « 2 ») | 9,9 / 8,9 | 10,6 | 15,7 |
| plain_test_mix | 30,3 | — (pas assez de « 2 ») | 7,1 / 7,7 | 9,7 | 16,1 |

Contrôle : `rp_s1_ema`, tirage 0, reproduit au bit près `comparison_local.csv` (plain 7,597 ; drs_p80 13,193 ; acceptation 0,4327 ; mêmes classes). Pour `bl_s1`, une interruption silencieuse du premier lancement a empêché l'écriture du CSV : ses lignes ont été récupérées du journal (`log_others.txt` → `*_bl_s1_fromlog.csv`) ; elles reproduisent aussi Noé (plain 201,04 ; drs_p80 182,51).

Variantes : `plain` sans tri ; `drs_pXX` DRS d'Azadi (`drs_azadi`) au percentile XX ; `platt` : D recalibré par d' = a·d + b ; `featlr` : dernière couche de D remplacée par une régression logistique sur fc3 ; `obrs_K3` : OBRS à budget 3 ; `drs_p80_plain_mix` : images du DRS p80, sous-échantillonnées pour avoir exactement le mélange de chiffres de `plain` ; `drs_p80_test_mix` et `plain_test_mix` : idem avec le mélange de MNIST test.

### 6.2 À relancer sur GPU (Juliet), par priorité
CPU de référence : un modèle, 10 variantes, 2 tirages ≈ 8 min sur 2 cœurs ; sur GPU, le coût est dominé par Inception (≈ 1 min par mesure de 10 000 images sur A100).

| # | Expérience | Pourquoi | Coût GPU | Commande |
|---|---|---|---|---|
| 1 | **DRS avec D recalibré (couche logistique sur fc3) sur RpGAN, mesuré en Inception, 3 seeds × 3 tirages** | trancher : la recalibration rend-elle le DRS utile au sens du classement ? | ≈ 20 min | copier `diag_drs.py`, remplacer `cnn_metrics` par `evaluate_prof(images, device, reference='test')` (déjà utilisé par `scripts/measure_runs.py`), lancer sur `rp_s{0,1,2}_ema` |
| 2 | **Même mesure sur la baseline commune (3 seeds)** | confirmer en Inception le gros gain vu au CNN avec D recalibré | ≈ 20 min | idem, modèles `bl_s{0,1,2}` |
| 3 | **Tirages multiples sur le protocole commun** | Noé n'a qu'un tirage par seed ; séparer le bruit du tirage du bruit d'entraînement | ≈ 15 min | dans `scripts/measure_runs.py`, `SEEDS = (0, 1, 2)` puis `PYTHONPATH=. python scripts/measure_runs.py outputs/comparison_local_manifest.csv results/comparison_local_3draws.csv` |
| 4 | **D entraîné en BCE contre G_EMA figé, à partir de zéro ou de D** (KeepTraining sur `G.pth` de RpGAN, arrêt sur val) | teste H1 + H2 ensemble, et la version « D entier » de la recalibration | ≈ 10 min par seed | `python keep_training.py --run runs/cmploc_rpgan_r1_ema_s1 --weights last` (charge `G.pth` = G_EMA), puis `python sweep.py --run runs/cmploc_rpgan_r1_ema_s1 --weights last_kt --sampling configs/sampling/drs_azadi.yaml --param percentile --values 50 80 95 --metric prof --reference test` — il faut d'abord ajouter `'last_kt': ('G.pth', 'D_last_kt.pth')` à `WEIGHT_FILES` dans `evaluate.py` (ligne 334 : seul `best_kt` existe) |
| 5 | **OBRS avec D recalibré, K = 2, 3, 5**, Inception | OBRS est l'acceptation optimale (Verine) ; voir s'il fait mieux que le DRS à budget égal | ≈ 15 min | `configs/sampling/obrs.yaml` avec `K` variable, même script que #1 |
| 6 | DRS sur le G brut de RpGAN (`G_raw.pth`) en Inception | isoler H2 | ≈ 5 min | manifeste avec `runs/cmploc_rpgan_r1_ema_s*/G_raw.pth,.../D.pth` puis `measure_runs.py` |

---

## 7. Approfondissements
- **Calibration de D en fiabilité** : tracer la courbe de fiabilité (diagramme réels/faux par tranche de σ(d)) et la statistique Z de Dawid (Turner, éq. 7) pour la baseline et RpGAN.
- **Pourquoi l'AUC de D tombe à 0,55** : R1 (γ = 1) lisse D et la perte relativiste n'impose pas de séparer les deux distributions, seulement de classer les paires ; à l'équilibre, la perte de D vaut 0,672, proche de log 2 = 0,693 (`logs/cmploc_rpgan_r1_ema_s1.log`, époque 200). Mesurer l'AUC au fil de l'entraînement.
- **Rejet dans l'espace latent** (Issenhuth 2021, latentRS) ou **MH-GAN** avec D calibré : mieux adaptés quand les variétés se recouvrent presque.
- **Rééquilibrage explicite des chiffres** (rejet guidé par le classifieur) : diagnostic utile, mais à ne pas présenter comme méthode (le classifieur sert aussi à la mesure).

---

## 8. Sources
- `wt-comparaison/` (branche `quentin/comparaison`) : `inference-drs.md`, `sampling.py` (l. 78-91, 187-213), `keep_training.py`, `configs/sampling/*.yaml`, `results/results.csv`, `results/results_prof.csv`, `results/comparison_local.csv`, `slides/comparaison.tex` (l. 136-160, 170-205), `slides/make_comparison_figures.py` (l. 232-276).
- `origin/noe/training`, commit 0a03c96 : `utils.py` (`D_train_rpgan`, `ema_update`), `train.py`, `scripts/compare_local.py`, `scripts/measure_runs.py`, `runs/cmploc_*/`, `logs/cmploc_*`.
- Papiers (`papiers/inference/`) : Azadi et al. 2019 (sections 3.1-3.2, 4.1-4.2, tableaux 1-2, annexe A) ; Turner et al. 2019 (sections 2, 4, figure 2) ; Verine et al. 2024 (sections 2.3, 3, 5.2, tableau 3) ; Issenhuth et al. 2021 (figure 2) ; Ansari et al. 2021 (tableau 3) ; Che et al. 2020.
- Diagnostics : `scratchpad/scripts/drs/diag_drs.py`, `scratchpad/results/drs/`.
