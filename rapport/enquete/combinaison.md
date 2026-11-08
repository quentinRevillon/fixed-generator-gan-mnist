# Combinaison, cohérence des chiffres et cahier des charges du rapport

Rédigé le 8/11/2026, en lecture seule (aucun fichier du dépôt ni du worktree modifié).
Scripts de recalcul : `scratchpad/scripts/combo/audit.py` et `scratchpad/scripts/combo/classes_tv.py`. Sortie brute : `scratchpad/findings/audit_output.txt`.
Légende : **[V]** = vérifié (recalculé depuis le fichier cité), **[H]** = hypothèse (plausible, non prouvée).

Quelques mots utilisés partout :
- **Seed** : graine du hasard. Deux runs qui ne diffèrent que par la seed donnent des scores différents. Cet écart s'appelle la *variance entre seeds* : c'est le bruit de fond.
- **Écart-type (σ)** : taille typique de ce bruit. On note « moyenne ± σ » sur les 3 seeds.
- **p (p-value)** : probabilité d'observer un écart au moins aussi grand si la méthode n'avait aucun effet. p < 0,05 : écart probablement réel. p > 0,2 : indiscernable du bruit.
- **Test apparié** : on compare la même seed d'entraînement avec et sans DRS (même G). C'est plus sensible qu'un test non apparié (Welch), qui compare deux groupes de runs indépendants.
- **Mise à jour (update)** : un pas d'optimisation sur un batch de 100 images. 200 époques sur 50 000 images = 100 000 mises à jour.
- **Protocole commun** : campagne de Noé (commit 0a03c96, `results/comparison_local.csv`) : mêmes 50k images, 200 époques, 3 seeds, dernière époque, mesure contre MNIST test.

---

## 0. Cahier des charges du rapport (énoncé, `enonce/diapos_enonce.pdf`)

| Point | Ce que dit l'énoncé | Page |
|---|---|---|
| Fichier | `report.pdf`, **11 novembre, 23:59** (heure de Paris), pour tous les groupes | 87 |
| Longueur | **5 pages de contenu maximum + une page de garde** du groupe. **Les annexes optionnelles ne sont pas lues.** Noms de fichiers et échéances imposés, **pénalités** sinon | 87 |
| Code | Dernières modifications de code : **12 novembre, 05:00** | 87 |
| Contrat du dépôt | `python generate.py --batch_size 256` charge le G de `checkpoints/`, produit **exactement 10 000 images 28×28 dans `samples/`** ; README qui documente entraînement **et** génération ; `requirements.txt` ; checkpoint accessible | 85 |
| Vérifiabilité | « Code quality itself is not graded; **your results and claims must be verifiable** » | 85 |
| Mesures attendues | « **FID, precision, recall and visual inspection** ». « **Explain what the measurements mean.** The theoretical PR curve and the finite-sample image estimator are different objects. » | 86 |
| Focus | « Focus on your experiments and interpretation, **not generic GAN background** ». « One method explained well beats a catalogue. » | 86, 89 |
| Méthode | 1. Reproduire une baseline (config, seed, checkpoint). 2. **Une hypothèse, un changement à la fois.** 3. **Protocole d'évaluation et budgets de calcul comparables.** 4. Ablations ; **séparer coût d'entraînement et coût de génération**. 5. Échantillons représentatifs + métriques, **y compris les échecs instructifs**. 6. **Interpréter le compromis : ce qui s'améliore, ce qui se dégrade, pourquoi.** « If possible, repeat runs to distinguish an improvement from seed variability. » | 84 |
| Organisation en 3 volets | « **Compare each intervention separately, then useful combinations.** » Ce n'est pas « trois mini-projets » | 76 |
| Inference (DRS) | « Keep the trained generator. **Report acceptance rates, total generation time and the full evaluation batch.** » | 82 |
| Training | « **Compare at a stated compute budget.** Extra gradients and additional updates are not free » (R1 = un gradient de plus) | 81 |
| Architecture | G fixe : couches, largeurs, dimensions latente et de sortie inchangées. D et la perte peuvent changer | 74, 80 |
| Plateforme | FID (Inception, toutes les images), precision et recall (VGG, k = 5, 9 984 images), inspection visuelle | 33 ; page de résultats |
| IA | Outils IA autorisés ; « you must verify and defend the code, results and writing » | 86 |

Rien d'autre n'est imposé (pas de plan type, pas de gabarit LaTeX). Le site du cours peut avoir ajouté des consignes : à vérifier.

**Score de la plateforme pour le modèle rendu** [V, page https://www.lamsade.dauphine.fr/~testplatform/prds-a2/results/last.html, mise à jour du 8/11 08:00] : commit fbdd93f1, **FID 21,59, precision 0,702, recall 0,585**, rang 24 (la page précise que le classement n'est pas une note). Notre mesure du même G (`cmploc_rpgan_r1_ema_s1`) : FID Inception 21,15, precision VGG 0,595, recall VGG 0,596 (`comparison_local.csv`). Le FID colle à 0,4 près ; la precision non (déjà constaté dans `baseline.md` §C : notre VGG n'est pas le leur).

**Points du cahier des charges non remplis aujourd'hui** [V] :
1. **`main` ne contient pas le code qui entraîne le modèle rendu.** `git show HEAD:train.py` ne contient ni `rpgan` ni `ema` ; seule `configs/baseline.yaml` existe. Le G rendu (`checkpoints/G.pth`, blob 66e370e = `runs/cmploc_rpgan_r1_ema_s1/G.pth` de `noe/training`) n'est donc pas reproductible depuis `main`. À fusionner avant le 12/11 05:00 (train.py, preprocessing.py, sampling.py, configs/compare_local, scripts/measure_runs.py).
2. Le README de `main` documente l'entraînement de la baseline, pas celui du modèle final.
3. Le rapport doit donner les **precision/recall de la plateforme (VGG)**. Les slides ne montrent que precision/recall du petit CNN. Les valeurs VGG existent déjà dans `comparison_local.csv` (colonnes `precision`, `recall`).
4. DRS : l'énoncé demande le **temps total de génération**. La slide l'a retiré (commit 4e32b8a « remove the CPU time »). À remettre dans le rapport (une colonne suffit).

---

## 1. Audit chiffre par chiffre des slides

Slides présentées : `main:slides/comparaison.tex` (= `slides.pdf`, commit bee5672). Numéros de ligne de ce fichier.
Sources : `WT` = `wt-comparaison/` ; `NOE` = `origin/noe/training` ; `NICO` = `origin/nico/preprocessing`.

### Slide 3 « Protocol » (l. 53-69)

| Ligne | Affiché | Recalculé / constaté | Source | Verdict |
|---|---|---|---|---|
| 59 | Baseline : 50k, 200 epochs, 1 seed, Inception+CNN, validation | 100 époques évaluées (2 à 200, une sur deux), référence `val` | `WT/runs/baseline_200ep_prof/metrics.jsonl` | OK |
| 60 | Sampling : baseline G, validation | 3 tirages du même G (pas 3 entraînements) | `WT/results/results.csv`, `results_prof.csv` | OK, préciser « 3 draws of the same G » |
| 61 | Data : 60k or fewer, **120k updates**, 3 to 8 seeds, CNN, test | Entraînés 120k, mais **tous les chiffres affichés sont lus à 60k mises à jour** (l. 112). Seeds stables : 3 à 5 (runs effondrés exclus) | `NICO/configs/data/data_ref.yaml` (200 époques), `figures/data/summary.csv` | **Écart** : écrire « read at 60k updates » |
| 62 | Training : 60k, 100 epochs, 3 seeds, CNN, test | 100 époques × 600 = 60k mises à jour | `NOE/configs/sweep/ema100_*.yaml` | OK |
| 63 | Common : 50k, 200 epochs, 3 seeds, Inception+CNN, test | 100k mises à jour, dernière époque | `WT/configs/compare_local/*.yaml`, `scripts/compare_local.py` | OK |
| 68 | Metrics : FID, precision, recall | Les P/R affichées sont celles du **petit CNN (k = 3)**, pas celles de la plateforme (VGG, k = 5) | `WT/evaluate.py` l. 1-7, 254-262 | **Ambigu** |

Manquent dans ce tableau, alors qu'ils changent les résultats : **EMA** (oui pour Training, non pour Baseline/Data, oui pour RpGAN seulement dans Common), **checkpoint retenu** (meilleur Inception sur val = dernière ; 60k fixe ; dernière ; dernière), **nombre de mises à jour** (100k ; 60k lus / 120k ; 60k ; 100k), **machine** (Juliet pour Data ; Mac M3 MPS pour Training et Common).

### Slide 4 « Baseline » (cmp_epochs.png, l. 71-75)

| Affiché | Recalculé | Source | Verdict |
|---|---|---|---|
| Epoch 50 : Inception 156, CNN 323 | 155,6 ; 323,5 | `metrics.jsonl` époque 50 | OK |
| Epoch 120 : 56 ; 75 | 56,2 ; 75,4 | idem | OK |
| Epoch 200 : 37 ; 169 | 37,0 ; 169,4 | idem | OK |
| « Epoch 200 kept: best Inception FID on validation » | min Inception = époque 200 = **dernière époque** | idem | OK, mais le minimum est au bord : Inception baissait encore, on ne sait pas s'il continuerait |

Remarque : la figure montre l'époque 120, alors que `cmp_two_measures.png` et `cmp_digits.png` montrent 106 (meilleur CNN). Le script prend 106 seulement si `G_ep106.pth` existe (`make_comparison_figures.py`, `plot_epochs`). Sans importance pour les chiffres.

### Slide 5 « Two FIDs » (cmp_two_measures.png, l. 77-81)

| Affiché | Recalculé | Verdict |
|---|---|---|
| Best Inception : epoch 200 | 200 (37,0) | OK |
| Best CNN : epoch 106 | 106 (55,7) | OK |

### Slide 6 « Digit proportions » (cmp_digits.png, l. 83-87)

| Affiché | Recalculé | Source | Verdict |
|---|---|---|---|
| Réel : 11 % (« 1 »), 9 % (« 5 ») | proportions standard de MNIST train (11,2 % et 9,0 %) | `data.mnist_split('train')` (non relancé : MNIST absent localement) | OK [H, valeurs connues de MNIST] |
| Époque 106 : 15 % (« 7 »), 6 % (« 6 ») | 15,4 % ; 5,9 % | `metrics.jsonl` époque 106, `classes` | OK |
| Époque 200 : 25 % (« 1 »), 2 % (« 2 ») | 24,7 % ; 2,3 % | époque 200 | OK |
| l. 86 « 98.7 % accuracy on MNIST test » | non recalculé (volet FID) | `WT/architecture-choix.md` l. 158 | non vérifié ici |

### Slides 7-10 « Data » (l. 89-131)

| Ligne / figure | Affiché | Recalculé | Source | Verdict |
|---|---|---|---|---|
| data_results.png | Réf. 76 / 0,75 / 0,72 | 75,7 / 0,753 / 0,718 (5 seeds stables) | `figures/data/summary.csv`, `runs.csv` | OK |
| idem | Noise 0.1 : 65 / 0,76 / 0,72 | 64,6 / 0,764 / 0,718 (4 seeds) | idem | OK |
| idem | Noise 0.3 : 68 / 0,76 / 0,72 | 67,7 / 0,763 / 0,719 (**3 seeds**) | idem | OK |
| idem | Sel. 95 % : 76 / 0,77 / 0,68 | 76,0 / 0,775 / 0,680 | idem | OK |
| idem | Sel. 90 % : 112 / 0,79 / 0,65 | 111,9 / 0,793 / 0,646 | idem | OK |
| idem | Sel. 50 % : 331 / 0,84 / 0,38 | 331,4 / 0,843 / 0,382 | idem | OK |
| l. 110 | « Selection: precision ↑, recall ↓, never beats the reference » | sel_95 = réf. (Welch p = 0,97) ; les autres font pire | `audit_output.txt` | OK |
| l. 111 | « Noise: same as the reference » | Welch p = 0,10 (σ = 0,1), p = 0,23 (σ = 0,3) | idem | OK |
| l. 126 | « 15 of 34 runs collapse without noise 0.3 » | réf. 3/8 + noise 0.1 4/8 + sel95 2/5 + sel90 2/5 + sel50 4/8 = **15/34** | `runs.csv` | OK (sur les 5 variantes affichées ; le rapport Data compte 19/52 sur toutes les variantes) |
| l. 127 | « Same rate with or without selection » | réf. 3/8 contre sélection 8/18, Fisher p = 1,0 | idem | OK |
| l. 128 + data_collapse.png | **« Noise 0.3: no collapse (0 of 8) »** | **Seulement 3 runs `noise_03` observés** (0/3). Les 5 autres étaient encore sur Juliet et ont été **comptés « non effondrés » d'office** : `EXPECTED = {'Noise 0.3': 8}` dans `NICO/figures/make_slide_figures.py` l. 5 et 30 | `NICO/figures/data/runs.csv` (3 lignes noise_03), `summary.csv` (`seeds = 3`) | **ÉCART MAJEUR** |
| l. 128 | « D wins less early » | interprétation, non mesurée dans les slides | — | [H] |

Signification statistique du 0/3 [V] : contre la référence 3/8, Fisher p = 0,49 ; contre les 15/34 autres runs, p = 0,26. Même le 0/8 affiché ne donnerait que p = 0,035 contre 15/34. Le rapport de Nicodème le dit lui-même (`rapport-donnée-illustré.md` l. 244 : « Le 0 sur 3 de `noise_03` […] ne prouve rien »). **Action :** rapatrier les 5 runs de Juliet ; sinon écrire « 0 of 3, not significant ».

### Slides 11-15 « Training » (l. 133-176)

| Ligne / figure | Affiché | Recalculé | Source | Verdict |
|---|---|---|---|---|
| l. 143 | γ = 1, EMA 0.999, 100 epochs | conforme | `NOE/configs/sweep/ema100_rpgan_r1_s0.yaml` | OK (mais le protocole commun en fait 200) |
| l. 151 | « 0.1: diverges » | FID 2233 à 2672 aux époques 10 et 20, D(G(z)) = 0,01 ; **runs tués à l'époque ~27** (code −15 = SIGTERM, `logs/sweep_training.log`) | `NOE/logs/ema100_rpgan_r1_g0.1_s*.log` | OK sur le fond ; préciser « stopped at epoch ~27 » |
| l. 152 | γ = 1 : FID 14,0 | 14,03 ± 0,88 | `WT/results.csv` | OK |
| l. 153 | γ = 10 : FID 17,7 | 17,69 ± 0,88 ; γ = 1 vs 10 : Welch p = 0,007 | idem | OK |
| l. 158 | « BCE: 1 run collapses » | seed 2 : FID 2770, 100 % de « 8 » | idem | OK |
| l. 161 | « Spread between seeds: ± 0.9 » | σ = 0,88 (demi-étendue 0,86) | idem | OK |
| cmp_loss_effect.png | parts des chiffres BCE (2 seeds) vs RpGAN (3 seeds) | BCE : 11,2 15,3 4,2 10,0 9,2 8,8 8,0 14,9 7,2 11,3 ; RpGAN : 10,2 12,2 9,6 10,9 9,6 9,6 9,3 12,3 8,1 8,2 | `WT/results.csv` `classes` | OK |
| cmp_training_results.png | BCE 58,7 / 0,80 / 0,79 ; RpGAN γ=1 14,0 / 0,72 / 0,80 ; R3GAN 14,4 / 0,70 / 0,80 ; γ=10 17,7 / 0,67 / 0,79 | identiques ; RpGAN vs R3GAN p = 0,63 (pas de différence) | idem | OK |
| l. 174 | « One BCE seed collapses (FID 2770) » | 2770,19 | idem | OK |
| l. 175 | 60k images, 100 epochs, EMA for all, test, 3 seeds | conforme | configs sweep | OK |

Attribution : `cmp_loss_effect.png` présenté (blob d9d50e5) est tracé par `slides/make_comparison_figures.py` **de `noe/training`** (fonction `plot_loss_effect` à la l. 154), pas par celui de `quentin/comparaison` (qui trace une autre figure, blob 5f32d58). L'en-tête l. 3-4 est donc inexact.

### Slides 16-19 « Sampling » (l. 178-202)

| Affiché | Recalculé (moyenne de 3 tirages) | Source | Verdict |
|---|---|---|---|
| Sans tri : Inception 36,7, CNN 169, P 0,862, R 0,741 | 36,69 ; 168,95 ; 0,862 ; 0,741 | `WT/results/results_prof.csv`, `results.csv` (run `baseline_200ep_prof`, `val`, `best`) | OK |
| p50 : 50 % gardés, 32,8 / 165 / 0,887 / 0,739 | 0,50 ; 32,77 ; 164,66 ; 0,887 ; 0,739 | idem | OK |
| p80 : 30 % gardés, 31,6 / 146 / 0,890 / 0,737 | 0,30 ; 31,59 ; 145,72 ; 0,890 ; 0,737 | idem | OK |
| p95 : 14 %, 31,3 / 121 / 0,890 / 0,739 | 0,14 ; 31,33 ; 121,13 | idem | OK |
| p99 : 5 %, 31,8 / 104 / 0,889 / 0,739 | 0,05 ; 31,79 ; 104,05 | idem | OK |
| l. 188 « γ ≈ −6.4 », d_M sur 10 000 images, ε = 1e-6 | ε et n_burnin = 10 000 conformes aux `params` ; γ non recalculé (volet DRS) | `params` du CSV | partiellement vérifié |

Vocabulaire : **γ désigne deux choses différentes** (poids de R1 l. 141-152, décalage du DRS l. 186-188). À renommer dans le rapport (par exemple λ pour R1, ou γ_DRS).

### Slides 20-22 « Common protocol » et « Conclusion » (l. 204-234)

| Affiché | Recalculé : moyenne ± σ (3 seeds) | Verdict |
|---|---|---|
| Baseline 38,7 / 181,8 / P 0,83 / R 0,75 | 38,66 ± 1,23 ; 181,83 ± 19,37 ; 0,83 ± 0,01 ; 0,75 ± 0,02 | OK |
| Baseline + DRS 34,4 / 154,0 / 0,87 / 0,74 | 34,36 ± 0,73 ; 153,96 ± **30,85** ; 0,87 ; 0,74 | OK |
| RpGAN + R1 + EMA 22,1 / 9,9 / 0,74 / 0,83 | 22,14 ± 1,26 ; 9,93 ± 2,47 ; 0,74 ; 0,83 | OK |
| + DRS 22,0 / 14,2 / 0,74 / 0,82 | 22,04 ± 1,10 ; 14,17 ± 2,28 | OK |
| + noise 0.3 21,4 / 9,0 / 0,74 / 0,82 | 21,44 ± 0,58 ; 9,01 ± 1,04 | OK |
| + noise 0.3 + DRS 21,9 / 13,9 / 0,75 / 0,82 | 21,93 ± 1,15 ; 13,90 ± 5,82 | OK |
| Conclusion l. 224-228 (mêmes valeurs) | identiques | OK |

Source : `WT/results/comparison_local.csv` (identique, aux fins de ligne près, à `NOE:results/comparison_local.csv`, commit 0a03c96). La colonne `seed` vaut 0 partout : c'est la seed d'échantillonnage. Les 3 seeds d'entraînement sont dans le nom du run (`_s0`, `_s1`, `_s2`).

**Bilan de l'audit** : tous les nombres affichés se retrouvent dans les fichiers, à l'arrondi près. Les écarts portent sur ce que les nombres veulent dire : le « 0 of 8 » (seul nombre faux), le « 120k updates », P/R CNN ou VGG, « diverges », l'attribution de figure, l'absence d'EMA dans les libellés.

---

## 2. Incohérences de protocole entre volets

| Dimension | Baseline | Sampling | Data | Training | Commun |
|---|---|---|---|---|---|
| Images d'entraînement | 50k (`mnist_split('train')`) | — | 60k (tout MNIST train) ; moins avec la sélection | 60k | 50k |
| Mises à jour lues | 100k (époque 200) | — | **60k** (sur 120k faites) | 60k (100 époques) | 100k |
| Perte | BCE (code fourni) | — | BCE | BCE ou RpGAN + R1 | BCE ou RpGAN + R1 |
| EMA de G | non | non | non | **oui pour tous** | **non pour la baseline, oui pour RpGAN** |
| Checkpoint | meilleur Inception sur val (= dernier) | idem | fixe à 60k (creux de la réf. sur test) | dernier | dernier |
| Référence | val | val | test | test | test |
| Inception mesuré | oui | oui | **non** | **non** | oui |
| P/R | CNN + VGG | CNN + VGG | CNN | CNN | CNN + VGG |
| Seeds | 1 | 1 G, 3 tirages | 3 à 8, effondrés exclus | 3 (BCE : 2 stables) | 3 |
| Machine | Juliet | — | Juliet (CUDA) | Mac M3 (MPS) | Mac M3 (MPS) |

**Conséquence : le mot « baseline » désigne trois modèles différents** [V] :
- CNN FID **182** (commun : BCE sans EMA, 100k mises à jour, test) ;
- **75,7** (Data : BCE sans EMA, lu à 60k, test, 5 seeds stables sur 8) ;
- **58,7** (Training : BCE **avec** EMA, 60k, test, 2 seeds stables sur 3).

L'écart 182 → 76 vient surtout du moment de lecture. Les courbes CNN sur val du protocole commun le montrent [V, `NOE:runs/cmploc_baseline_s*/metrics.jsonl`] : la baseline atteint son creux vers les époques 100 à 120 (FID 50, 65, 63), puis remonte jusqu'à 159, 196 et 179 à l'époque 200. C'est le même phénomène que la dérive vers les « 1 » décrite dans le rapport Data (l. 210). RpGAN + EMA, lui, baisse encore à l'époque 200 (11, 6, 9).

**Comparaisons légitimes** (même protocole) :
- les 6 lignes du protocole commun entre elles (seule source pour Inception + VGG + CNN sur test) ;
- dans le volet Training : BCE+EMA vs RpGAN+R1+EMA vs R3GAN vs γ (même budget, EMA pour tous) ;
- dans le volet Data : variantes entre elles, à 60k mises à jour ;
- dans le volet DRS : percentiles entre eux sur le même G.

**Comparaisons illégitimes** :
- un chiffre du volet Data (76) avec un chiffre du volet Training (58,7) ou du commun (182) : lecture, EMA, machine et nombre d'images diffèrent ;
- des gains en % d'un volet à l'autre (« la perte divise le FID par 4, la sélection par… ») ;
- placer Data ou Training sur l'échelle de la plateforme : Inception n'y a pas été mesuré ;
- le CNN FID de la baseline à l'époque 200 comme mesure « juste » de la baseline : il la pénalise, car la baseline est lue après son creux CNN. L'écart RpGAN vs baseline reste énorme avec Inception (−16,5, p < 0,001), mais le rapport doit dire que le facteur ×18 en CNN FID est gonflé par ce choix d'époque ;
- val (Baseline, Sampling) contre test (le reste) : écart faible en pratique (Inception 37,0 sur val pour le G de `baseline_200ep_prof` ; 38,7 ± 1,2 sur test pour 3 nouvelles seeds), mais à signaler.

**Conclusion l. 219 « The loss gives the largest gain »** : au protocole commun, la perte **et** l'EMA changent ensemble. On ne peut pas les séparer là. Seul le volet Training sépare la perte (EMA pour tous) : 58,7 → 14,0 en CNN FID, à 60k mises à jour. Une estimation de l'effet de l'EMA seule demande de croiser deux campagnes : BCE sans EMA lu à 60k (Data, 75,7) contre BCE avec EMA à 60k (Training, 58,7). Les deux ont les mêmes données et le même nombre de mises à jour, mais pas la même machine [H]. Formulation juste : « RpGAN + R1 (avec EMA) donne le plus grand gain ; dans le volet Training, où l'EMA est appliquée à tous, la perte seule fait passer le CNN FID de 58,7 à 14,0. »

**Choix du modèle rendu** : seed 1, choisie sur le CNN FID de validation (10,8 ; 6,0 ; 9,1, d'après le message du commit fbdd93f). C'est correct : on n'a pas choisi sur test. Mais le FID test de cette seed (21,15) est le meilleur des trois. Le rapport doit annoncer la moyenne sur 3 seeds (22,1 ± 1,3) comme performance attendue de la méthode, et le 21,6 de la plateforme comme score du modèle rendu.

---

## 3. Combinaison

### Ce qui a été combiné [V]
Au protocole commun : {BCE sans EMA, RpGAN + R1 + EMA, RpGAN + R1 + EMA + bruit 0,3} × {sans DRS, DRS percentile 80}, 3 seeds chacun. Le DRS utilise le D final du run, sans KeepTraining. **La sélection n'a jamais été combinée.**

### Tableau complet (moyenne ± σ, 3 seeds, test) — à mettre dans le rapport

| Config | FID Inception | P VGG | R VGG | FID CNN | P CNN | R CNN | Acceptation | Écart au mélange réel (TV) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 38,7 ± 1,2 | 0,47 | 0,51 | 182 ± 19 | 0,83 | 0,75 | 1 | 0,244 |
| Baseline + DRS | 34,4 ± 0,7 | 0,51 | 0,49 | 154 ± 31 | 0,87 | 0,74 | 0,30 | 0,225 |
| RpGAN + R1 + EMA | 22,1 ± 1,3 | 0,59 | 0,59 | 9,9 ± 2,5 | 0,74 | 0,83 | 1 | 0,044 |
| + DRS | 22,0 ± 1,1 | 0,59 | 0,58 | 14,2 ± 2,3 | 0,74 | 0,82 | 0,43 | 0,054 |
| + bruit 0,3 | 21,4 ± 0,6 | 0,60 | 0,61 | 9,0 ± 1,0 | 0,74 | 0,82 | 1 | 0,043 |
| + bruit 0,3 + DRS | 21,9 ± 1,2 | 0,60 | 0,60 | 13,9 ± 5,8 | 0,75 | 0,82 | 0,42 | 0,058 |

TV (variation totale) = moitié de la somme des écarts absolus entre la part de chaque chiffre générée et celle de MNIST test. 0 = mélange parfait. Calcul : `classes_tv.py`.

### Les écarts dépassent-ils le bruit ? [V, `audit_output.txt`]

| Effet | Δ FID Inception | Δ FID CNN | Δ P / R (VGG) | Verdict |
|---|---|---|---|---|
| Perte + EMA (baseline → RpGAN) | −16,5 (Welch p < 0,001) | −172 (p = 0,004) | +0,12 / +0,08 (p ≤ 0,007) | **réel, très grand** |
| Bruit 0,3 sur RpGAN (22,1 → 21,4) | −0,7 (p = 0,46) | −0,9 (p = 0,60) | +0,01 / +0,02 (p = 0,09 ; 0,18) | **indiscernable du bruit** |
| DRS sur baseline | −4,3 (apparié p = 0,06) ; les 3 seeds baissent (−6,0 ; −2,2 ; −4,7) | −28 (p = 0,24 ; −4, −19, −61) | **+0,04 (p = 0,004) / −0,01 (p = 0,012)** | gain Inception probable ; **compromis precision ↑ / recall ↓ net** ; CNN non significatif |
| DRS sur RpGAN | −0,1 (p = 0,84) | **+4,2 (apparié p = 0,033 ; +4,3 ; +5,6 ; +2,9)** | 0 / −0,01 | **neutre en Inception, nuisible en CNN** |
| DRS sur RpGAN + bruit | +0,5 (p = 0,52) | +4,9 (p = 0,31) | 0 / 0 | même tendance, non significative |

**Taille minimale détectable** avec 3 seeds par groupe (test bilatéral 5 %, puissance 80 %) : environ (2,78 + 0,94) × σ × √(2/3), soit **≈ 3,6 points de FID Inception** (σ ≈ 1,2) et **≈ 7,6 points de CNN FID** (σ ≈ 2,5). Tout effet plus petit, comme celui du bruit, est invisible avec ce budget. Pour détecter 0,7 point il faudrait environ 45 seeds par groupe : ce n'est pas à poursuivre.

### Les effets s'additionnent-ils ? Non.
- **DRS × perte : interaction négative** [V]. Le DRS gagne 4,3 points d'Inception sur la baseline et 0 sur RpGAN. Il fait même perdre 4 points de CNN FID sur RpGAN. Le gain du DRS dépend donc du générateur : plus G est déjà bon, moins D a de « mauvais » échantillons à rejeter.
- Indices compatibles [V] : (a) au même percentile 80, le DRS garde 43 % des images de RpGAN contre 30 % pour la baseline, signe que D distingue moins bien les bons et les mauvais échantillons ; (b) le DRS dégrade légèrement le mélange des chiffres de RpGAN (TV 0,044 → 0,054), et le CNN FID est très sensible à ce mélange ; (c) sur la baseline, le DRS fait l'inverse (TV 0,244 → 0,225).
- Explications possibles [H, à confirmer par le volet DRS] : (1) **G est la moyenne EMA, mais D a été entraîné contre le G brut** : son score n'estime pas le rapport de densités du G évalué ; (2) le D de RpGAN est **relativiste** : seules les différences D(réel) − D(faux) ont un sens, et son optimum n'est pas le log-rapport de densités sur lequel repose le DRS (Azadi et al.) ; (3) R1 lisse D près des données, donc ses scores sont moins tranchés.
- **Bruit × perte** : le bruit sert à éviter l'effondrement (D qui gagne trop tôt). Or RpGAN + R1 ne s'effondre déjà jamais (0/9 runs au protocole commun, 3/3 stables dans le volet Training). Il n'y a donc rien à corriger, et aucun effet n'est attendu ni observé [V pour l'observation, H pour la cause].
- **Sélection** : jamais combinée. Elle n'a jamais battu la référence seule (sel_95 = réf.), donc aucun gain n'est attendu [H].

### Ce qui manque pour une analyse factorielle propre
Facteurs : perte (BCE / RpGAN+R1), EMA (non / oui), bruit (0 / 0,3), sélection (100 % / 90-95 %), DRS (non / oui). Le DRS ne coûte rien à ajouter : il est mesuré sur chaque G. Cellules manquantes au protocole commun, par ordre d'utilité :
1. **BCE + EMA** : sépare perte et EMA, donc justifie directement la phrase de conclusion.
2. **BCE + bruit 0,3** : le seul cas où le bruit peut agir (la baseline s'effondre parfois). Les 5 seeds `noise_03` déjà lancées sur Juliet (volet Data) répondent à la question de l'effondrement sans rien relancer.
3. **RpGAN + R1 sans EMA** : effet de l'EMA sur RpGAN.
4. RpGAN + R1 + EMA + sélection 90 % : complète la combinaison des trois volets. Gain attendu nul.

---

## 4. Structure proposée du rapport (5 pages + page de garde, annexes non lues)

Gabarit à reprendre pour chaque méthode, en 4 lignes : **Attendu (papier)** / **Observé (chiffre + p)** / **Pourquoi** / **Quand ça ne marche pas**.

| # | Partie | Questions auxquelles elle répond | Tableaux / figures | Longueur |
|---|---|---|---|---|
| 1 | **Introduction et protocole** | Quel problème (G fixe, 3 leviers) ? Quelle hypothèse par levier ? Comment compare-t-on (split 50k/10k/10k, 10 000 images, test, 3 seeds, moyenne ± σ, test apparié) ? Quel budget (100k mises à jour, coût R1 +5-10 %/époque) ? | Tableau de protocole **corrigé** (avec mises à jour lues, EMA, checkpoint, machine), cf. §2 | 0,6 p |
| 2 | **Évaluation (FID)** | Que mesurent FID, precision et recall (définitions courtes, P/R k-NN = estimateur fini, pas la courbe PR théorique) ? Pourquoi un 2ᵉ FID (CNN MNIST) en plus d'Inception (remarque du prof) ? Les deux choisissent-ils la même époque ? Notre mesure reproduit-elle la plateforme (FID oui à ~0,5 près, P non) ? | `cmp_two_measures.png` + `cmp_digits.png` côte à côte (une figure) ; mini-tableau « notre mesure vs plateforme » (21,15 vs 21,59 ; P 0,60 vs 0,70) ; plafond réel train vs test (CNN 5,1) | 0,7 p |
| 3 | **Données** (Nicodème) | La sélection d'instances améliore-t-elle precision au prix du recall ? Le bruit d'instance stabilise-t-il ? Pourquoi les runs BCE s'effondrent-ils (sigmoïde saturée) ? | `data_results.png` ; nombre d'effondrements **corrigé** (0/3 ou 0/8 réel) ; gabarit ×2 (sélection, bruit) | 0,9 p |
| 4 | **Entraînement** (Noé) | Que change la perte relativiste + R1 ? Pourquoi R1 est-il indispensable (RpGAN seul diverge) ? Choix de γ ? Effet sur le mélange des chiffres ? | `cmp_training_results.png` ou tableau ; `cmp_loss_effect.png` ; gabarit | 0,9 p |
| 5 | **Inference** (Quentin) | Le DRS améliore-t-il precision sans perdre de recall ? Quel coût (taux d'acceptation, **temps total de génération**) ? Pourquoi échoue-t-il sur RpGAN + EMA (remarque du prof) ? | `cmp_sampling_results.png` + colonne temps ; `cmp_kept_rejected.png` (petit) ; gabarit | 0,8 p |
| 6 | **Combinaison** | Les gains s'additionnent-ils ? Quels écarts dépassent le bruit ? | Tableau du §3 (Inception + **P/R VGG** + CNN + acceptation, ± σ) ; `cmp_combined_samples.png` ; phrase sur la taille minimale détectable | 0,8 p |
| 7 | **Conclusion** | Modèle rendu et score plateforme (21,59 / 0,702 / 0,585) ; ce qui marche (perte + EMA), ce qui ne marche pas (sélection, DRS sur un bon G, bruit sans effet mesurable) ; limites (3 seeds, Mac vs Juliet, cellules manquantes) | — | 0,3 p |

Total ≈ 5,0 p. Pour tenir : 5 à 6 figures au plus, toutes sur la largeur avec légende courte ; pas de rappel générique sur les GAN (consigne p. 86) ; les papiers en une ligne chacun.

Textes « attendu (papier) » à rédiger [références] : DeVries et al. 2020 (sélection : meilleure qualité, moins de couverture) ; Sønderby et al. 2017 (bruit d'instance : supports qui se recouvrent, D moins tranchant) ; Huang et al. 2024, R3GAN (RpGAN + R1 + R2 : convergence locale, moins d'oubli de modes) + Mescheder et al. 2018 pour R1 ; Azadi et al. 2019 (DRS : corrige p_G avec D optimal, suppose D ≈ log-rapport de densités) ; EMA (Yazıcı et al. 2019) à citer, car elle fait partie de la méthode retenue.

---

## 5. Corrections transverses des slides (`main:slides/comparaison.tex`)

Utiles aussi pour le rapport, qui reprendra ces phrases.

| Ligne | Actuel | Correction proposée | Raison |
|---|---|---|---|
| 3-4 | figures tracées par `make_comparison_figures.py` (quentin/comparaison) sauf data_* | « cmp_loss_effect.png : make_comparison_figures.py de noe/training ; drs_paper.png et r3gan_paper.png : captures des papiers » | attribution inexacte (§1) |
| 48 | « Relativistic loss (RpGAN + R1) » | « RpGAN + R1, with EMA of G » | l'EMA fait partie de la méthode comparée, mais elle n'est nommée qu'en l. 143 et 226 |
| 59-63 | tableau de protocole | ajouter les colonnes « Updates read » (100k / – / 60k of 120k / 60k / 100k) et « EMA » (no / – / no / all / RpGAN only) ; ligne Data : « read at 60k updates » | §2 |
| 68 | « Metrics: FID, precision, recall » | « FID (Inception and small CNN); precision, recall (small CNN, k = 3; platform: VGG, k = 5) » | les P/R affichées ne sont pas celles de la plateforme |
| 128 | « Noise 0.3: no collapse (0 of 8) » | « Noise 0.3: 0 of 3 collapsed (not significant, Fisher p = 0.49) », ou le vrai compte une fois les 5 runs rapatriés | **5 runs sur 8 non observés** (§1) |
| 141 / 186 | γ pour R1 et pour le DRS | garder γ pour R1, noter le décalage du DRS autrement (γ_DRS ou c) | même symbole, deux sens |
| 151 | « 0.1: diverges » | « 0.1: diverges (FID > 2000 at epoch 20, runs stopped) » | runs arrêtés à l'époque ~27 |
| 201 | « against validation, 3 draws » | « against validation, 3 draws of the same G » | ne pas confondre avec 3 seeds d'entraînement |
| 200 | « Higher percentile: fewer images kept » | « Inception FID stops improving after p80; precision +0.03, recall unchanged » | message trivial ; c'est le compromis P/R qui compte |
| 207 | « The loss improves both FIDs. DRS only helps the baseline. » | « RpGAN + R1 + EMA improves both FIDs. DRS: −4.3 Inception on the baseline (all 3 seeds); no gain on RpGAN, +4.2 CNN FID (p = 0.03). » | nuance statistique |
| 208 | « Same split, 200 epochs… » | ajouter « last epoch; EMA only for RpGAN » | protocole |
| 219 | « The loss gives the largest gain. » | « RpGAN + R1 (with EMA) gives the largest gain. » | perte et EMA confondues au protocole commun (§2) |
| 228 | « + noise 0.3 21.4 / 9.0 » | ajouter « (n.s.) » ou les ± σ dans le tableau | 21,4 vs 22,1 : p = 0,46 |
| 231 | « Noise 0.3: fewer collapses. » | « Noise 0.3: no measurable effect (0 of 3 collapsed; RpGAN never collapses) » | non démontré |
| 232 | « Proposed final model: RpGAN + R1 + EMA. » | ajouter « submitted: seed 1, platform FID 21.6, P 0.70, R 0.59 » | lien avec la plateforme |

Vocabulaire à fixer dans le rapport : « baseline » (préciser laquelle), « noise » = *instance noise*, « collapse » (le rapport Data montre que c'est une **saturation de D** avec une sortie de bruit identique pour tous les z, pas un mode collapse sur un chiffre : `rapport-donnée-illustré.md` l. 223), « seed » (d'entraînement ou d'échantillonnage), « draws ».

---

## 6. Expériences à relancer (Juliet)

Hypothèse de coût : 2 h à 2 h 24 par run de 200 époques sur Mac M3 [V, `NOE:logs/cmploc_*_s0.log`] ; sur un GPU de Juliet, moins d'une heure par run [H] (`compare.sh` réserve 02:00:00). Échéance du rapport : 11/11 23:59. Ne lancer que P0 et P1.

| Prio | Expérience | Pourquoi | Coût | Commande |
|---|---|---|---|---|
| **P0** | Rapatrier les 5 seeds `noise_03` / `noise_03_e100` encore sur Juliet | rend vrai ou faux le « 0 of 8 » ; aucun calcul | 0 GPU-h | `rsync -av --include='*/' --include='metrics.jsonl' --include='grid.png' --exclude='*' juliet:<repo>/runs/noise_03* runs/` puis `python figures/make_slide_figures.py` (NICO), avec `EXPECTED = {}` |
| **P1** | **BCE + EMA**, protocole commun, 3 seeds | sépare perte et EMA (conclusion) | 3 runs ≈ 3 GPU-h | voir ci-dessous |
| **P1** | **BCE + bruit 0,3**, protocole commun, 3 seeds (8 si possible pour compter les effondrements) | seul cas où le bruit peut agir | 3 à 8 runs | idem |
| P2 | RpGAN + R1 **sans EMA**, 3 seeds | effet de l'EMA sur RpGAN | 3 runs | idem |
| P2 | DRS sur RpGAN avec **D réentraîné contre le G EMA** (`keep_training.py`, sans réentraîner G) | teste l'hypothèse (1) de §3 sur l'échec du DRS | minutes | voir le volet DRS |
| P3 | RpGAN + R1 + EMA + sélection 90 % (220 époques pour garder 100k mises à jour, `anneal_epochs` inchangé) | combinaison des 3 volets ; gain nul attendu | 3 runs | idem |

Configs proposées (à créer dans `configs/compare_juliet/`, copies de `configs/compare_local/baseline.yaml` où l'on change seulement `name` et la section indiquée ; `eval.prof: false` pour garder le même protocole que la campagne locale) :
- `baseline_ema.yaml` : `name: cmp2_baseline_ema` + `ema: {decay: 0.999}`
- `baseline_noise_03.yaml` : `name: cmp2_baseline_noise_03` + `data: {instance_noise: {enabled: true, sigma: 0.3, anneal_epochs: 100}}`
- `rpgan_r1.yaml` : copie de `compare_local/rpgan_r1_ema.yaml` sans la section `ema`, `name: cmp2_rpgan_r1`

Lancement : copier `scripts/compare.sh` en remplaçant `configs/compare/` par `configs/compare_juliet/` et `run=cmp_…` par `run=$(grep '^name:' "$config" | cut -d' ' -f2)_s$seed` (train.py ajoute lui-même `_s<seed>`). Dans le job de mesure, **mesurer `G.pth` / `D.pth` (dernière époque) et non `G_best.pth`**, comme `compare_local.py` :
```bash
for s in 0 1 2; do for c in baseline_ema baseline_noise_03 rpgan_r1; do
  echo "cmp2_$c,runs/cmp2_${c}_s$s/G.pth" >> outputs/cmp2_manifest.csv
  echo "cmp2_$c + DRS,runs/cmp2_${c}_s$s/G.pth,runs/cmp2_${c}_s$s/D.pth" >> outputs/cmp2_manifest.csv
done; done
PYTHONPATH=. python scripts/measure_runs.py outputs/cmp2_manifest.csv results/comparison_juliet2.csv
```
Limite à écrire dans le rapport : ces runs tourneront sur CUDA (Juliet), alors que la campagne commune a tourné sur MPS (Mac). Contrôle conseillé : relancer aussi `cmploc_baseline` seed 0 sur Juliet, pour vérifier que la machine ne déplace pas les chiffres (attendu : Inception ≈ 38,7 ± 1,2).

---

## Sources

- Énoncé : `enonce/diapos_enonce.pdf` p. 33, 73-89 (texte extrait : `scratchpad/enonce.txt`) ; `enonce/datalab-slurm.pdf` (rien sur le rapport).
- Plateforme : https://www.lamsade.dauphine.fr/~testplatform/prds-a2/results/last.html (consultée le 8/11, mise à jour du 8/11 08:00).
- Slides : `main:slides/comparaison.tex`, `main:slides/figures/*.png` ; scripts : `WT/slides/make_comparison_figures.py`, `NOE:slides/make_comparison_figures.py` (`plot_loss_effect`), `NICO:figures/make_slide_figures.py`.
- Données : `WT/results/comparison_local.csv` (= `NOE`, commit 0a03c96) ; `WT/results.csv` (campagne Training) ; `WT/results/results.csv`, `WT/results/results_prof.csv` (DRS) ; `WT/runs/baseline_200ep_prof/metrics.jsonl` ; `WT/figures/data/summary.csv`, `runs.csv` (= `NICO`) ; `NOE:runs/cmploc_*/metrics.jsonl` ; `NOE:logs/ema100_rpgan_r1_g0.1_s*.log`, `logs/sweep_training.log`, `logs/cmploc_*.log`.
- Configs : `WT/configs/compare_local/*.yaml`, `WT/configs/compare/*.yaml`, `NICO:configs/data/*.yaml`, `NOE:configs/sweep/*.yaml`.
- Rapports : `WT/rapport-donnée-illustré.md` (l. 51, 76, 144, 196, 210, 223, 244), `WT/training-r3gan.md` (§4-5), `WT/architecture-choix.md` (l. 142-158), `WT/baseline.md` §C.
- Git : `git log --all` et `git ls-remote origin` : **aucun commit après fbdd93f (4/11 22:15:52)** ; les refs distantes égalent les refs locales. `checkpoints/G.pth` de `main` = blob 66e370e = `NOE:runs/cmploc_rpgan_r1_ema_s1/G.pth`.
