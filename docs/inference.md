# Volet Inférence : rejet optimal à budget fixé (OBRS)

**Responsable :** Quentin. Code : `sampling.py`.

**Question de recherche.** À budget de calcul fixé, un échantillonnage piloté par le discriminateur augmente-t-il la précision sans coûter de rappel ? Corrige-t-il le déséquilibre des classes du générateur ?

**Sources :**
- slides du cours 26-28 (courbe PR), 35-37 (truncation), 59-71 (amélioration d'un modèle entraîné), 73-89 (consignes A2) ;
- les 8 papiers de `inference papers/`, lus en entier.

---

## 1. Pourquoi OBRS, et pourquoi DRS et la truncation comme témoins

| Méthode | Principe | Besoins | Coût / image | Verdict |
|---|---|---|---|---|
| **OBRS** (Verine et al. 2024) | rejet optimal à budget K : $a(x)=\min(1, e^{d(x)-\tau_K})$ | D (logit), **aucun entraînement** | K passes de G et D | **méthode étudiée** |
| DRS (Azadi et al. 2019) | rejet avec M estimé et un décalage γ heuristique | idem | K (γ calibré pour le même K) | témoin à coût égal |
| Truncation (Brock et al. 2019) | z tronqué à $[-\psi, \psi]^{100}$ | rien | 1 | témoin « sans D » |
| DGflow / DDLS (Ansari 2021, Che 2021) | gradient de d(G(z)) dans l'espace latent | gradients de D∘G | ≈ 3N passes | extension possible (raffiner puis rejeter) |
| MH-GAN (Turner 2019) | chaîne de Metropolis–Hastings sur $e^{d}$ | D calibré, vraies images au démarrage | K passes | cité ; moins bon qu'OBRS à budget égal (Table 2 d'OBRS) |
| Latent reweighting (Issenhuth 2021) | réseau w(z) appris, rejet dans l'espace latent | critique WGAN-GP à entraîner | ≈ 1 | cité ; recoupe le volet Entraînement |
| DOT (Tanaka 2019) | transport optimal latent guidé par D | D spectralement normalisé de préférence | ≈ 3N passes | cité ; gain mal établi avec un D en BCE |

**Arguments pour OBRS :**
- c'est le papier du prof (slides 63-66) ;
- sa théorie se teste sur nos chiffres : la Prop. 3.3 prédit précision ↑ et rappel conservé ;
- aucun entraînement, moins d'une seconde pour 10 000 images sur GPU ;
- conforme à la slide 82 : « Keep the trained generator. Report acceptance rates, total generation time ».

**Arguments pour les témoins :**
- la slide 57 compare justement λ-PR à la truncation ;
- la slide 66 compare OBRS à DRS.

## 2. Ce qui est implémenté

```
sampling.py         un seul fichier, en 4 sections :
  latents           gaussian_latents, truncated_latents (inversion exacte), sample_plain, sample_truncation
  ratio de densités d_logit(D, x) = log r(x) (logit avant sigmoïde), propose(G, D, n, rng)
  rejet             obrs_threshold, obrs_accept_prob, drs_gamma, drs_accept_prob, rejection_sample,
                    sample_obrs, sample_drs
  registre          SAMPLERS = {plain, truncation, drs, obrs} ; sample(G, D, n, cfg, seed)
configs/sampling/   plain.yaml, truncation.yaml, drs.yaml, obrs.yaml
```

- **Contrat.** `sample_<method>(G, D, n, rng, batch_size, **params) -> (x, stats)` renvoie exactement n images dans [-1, 1]. `stats` contient au moins `n_proposals` et `acceptance_rate`, et `sample()` ajoute la durée.
- **OBRS en pratique** (Thm 3.2) :
  1. Tirer un pool de 50 000 propositions et calculer leurs logits $d(x)$.
  2. Trouver $\tau_K$ par bissection pour que $\text{mean}(\min(1, e^{d-\tau_K})) = 1/K$, plafonné à $\max d$ (si $K \geq M$, on retombe sur le rejet exact).
  3. Rejeter avec $a_{\rm OBRS}$ sur des propositions fraîches.
- **DRS** (slides Def. 3.1) : $a = r / (r(1-e^\gamma) + M e^\gamma)$, avec M le max du pool et γ calibré pour le même taux d'acceptation 1/K.
- **Truncation** : $z = \Phi^{-1}(u)$, avec $u \sim U(\Phi(-\psi), \Phi(\psi))$. C'est exactement « rééchantillonner chaque $|z_i| > \psi$ », sans boucle.
- **Hasard.** Tout le hasard vient d'un `torch.Generator` sur CPU : même seed ⇒ mêmes images sur tout device.
- **Tests** (`tests/test_sampling.py`) :
  - n images exactement, déterminisme ;
  - budgets 1/K respectés ;
  - cas limites : K = 1 accepte tout, K > M donne le rejet exact, γ = 0 donne le rejet exact ;
  - variance de la normale tronquée.

## 3. Protocole

- Chaque réglage est évalué sur **3 seeds** (10 000 images chacune) avec FID, P et R k-NN (k = 3), plus l'histogramme des classes, le taux d'acceptation, le nombre de propositions et le temps.
- **Choix des réglages (K, ψ) sur `val`. Chiffres finaux sur `test`** (`--reference test`).
- OBRS et DRS sont comparés **au même taux d'acceptation** (même coût).
- Commande unique sur Juliet : `scripts/sweep.sh runs/baseline_v1 best val`, puis la même chose avec `test`.

## 4. Résultats préliminaires

Ils sont obtenus sur le modèle actuel de `checkpoints/` (ancien run, 3 seeds, référence `val`, CPU). Ils seront refaits sur `baseline_v1` une fois celle-ci entraînée sur Juliet.

Données brutes : `results/results.csv` (63 lignes). Figures : `figures/val/`. Moyenne ± écart-type sur 3 seeds, 10 000 images par évaluation.

| Méthode | Réglage | Acceptation | FID ↓ | Précision ↑ | Rappel ↑ | Temps (CPU) |
|---|---|---|---|---|---|---|
| plain | — | 1 | 97,6 ± 3,6 | 0,834 ± 0,004 | 0,744 ± 0,003 | 0,2 s |
| **OBRS** | K = 2 | 0,50 | **77,5 ± 1,2** | 0,853 ± 0,002 | 0,744 ± 0,005 | 3,2 s |
| DRS | K = 2 | 0,50 | 80,0 ± 2,3 | 0,847 ± 0,001 | 0,748 ± 0,001 | 2,9 s |
| **OBRS** | K = 5 | 0,20 | **57,8 ± 2,7** | 0,852 ± 0,000 | 0,745 ± 0,002 | 4,1 s |
| DRS | K = 5 | 0,20 | 64,2 ± 1,8 | 0,850 ± 0,002 | 0,750 ± 0,005 | 4,1 s |
| **OBRS** | K = 10 | 0,10 | **48,8 ± 3,2** | 0,853 ± 0,002 | 0,746 ± 0,000 | 6,1 s |
| DRS | K = 10 | 0,10 | 53,5 ± 3,8 | 0,853 ± 0,003 | 0,751 ± 0,003 | 6,1 s |
| OBRS / DRS | K = 50 | 0,02 | 47,1 ± 1,8 / 45,8 ± 1,6 | 0,855 / 0,855 | 0,740 / 0,742 | 20 s |
| truncation | ψ = 3 | 1 | 95,3 ± 1,0 | 0,835 | 0,743 | 0,2 s |
| truncation | ψ = 2 | 1 | 133,8 ± 1,3 | 0,857 | 0,697 | 0,3 s |
| truncation | ψ = 1 | 1 | 598,1 ± 6,4 | 0,907 | **0,349** | 0,2 s |

Part de chaque chiffre 0-9 (%), et écart à la répartition uniforme (variation totale) :

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | écart |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 10,8 | **18,7** | **3,4** | 8,0 | 10,1 | 7,0 | 7,6 | 14,5 | 5,3 | **14,5** | 18,7 % |
| OBRS, K = 10 | 8,6 | 11,4 | 6,9 | 10,2 | 9,3 | 11,6 | 7,4 | 17,8 | 8,1 | 8,8 | **11,0 %** |
| DRS, K = 10 | 8,8 | 12,8 | 6,1 | 10,0 | 9,4 | 10,6 | 7,3 | 17,9 | 7,5 | 9,6 | 11,3 % |
| truncation, ψ = 1,5 | 9,3 | 16,4 | 2,0 | 8,3 | 6,8 | 7,0 | 5,8 | 13,2 | 5,0 | 26,1 | 25,8 % |

**Lecture :**

1. **Le rejet améliore la précision sans coûter de rappel (H1).**
   - La précision passe de 0,834 à environ 0,853 dès K = 1,5. Le rappel reste à 0,744 ± 0,005 à tous les budgets.
   - C'est le comportement que prédit la Prop. 3.3 (α ↑, β conservé).
   - Mais la précision **sature vite**, bien en dessous du plafond de 0,90. L'estimateur k-NN est un indicateur de support, et D n'est pas optimal.
2. **Le FID, lui, continue de baisser avec K** : 97,6 → 48,8 à K = 10, soit −50 %. Il faut expliquer ce gain, puisque la précision ne bouge plus :
   - le rejet **corrige la répartition des classes**, qui passe d'un écart de 18,7 % à 11,0 % de l'uniforme ;
   - les « 1 » et « 9 » sur-produits sont rejetés, les « 2 » doublent (3,4 → 6,9 %).
   - Le gain mesuré par le FID vient donc surtout de la **forme de la distribution**, pas de la qualité image par image.
   - Limite : les « 5 » et « 7 » deviennent un peu sur-représentés, car D ne connaît pas les classes.
3. **OBRS bat DRS à coût égal (H2), sauf aux extrêmes.**
   - Les deux méthodes voient, pour une même seed, les mêmes propositions et les mêmes uniformes. La comparaison se fait donc **par paires**.
   - Écart FID(DRS) − FID(OBRS) par seed :

     | K | 1,5 | 2 | 3 | 5 | 10 | 20 | 50 |
     |---|---|---|---|---|---|---|---|
     | seed 0 / 1 / 2 | −0,1 / +0,2 / −3,5 | +4,0 / +1,6 / +1,8 | +6,8 / +5,5 / +3,3 | +8,4 / +5,8 / +4,9 | +3,7 / +5,2 / +5,3 | +1,6 / +2,3 / +4,3 | −1,1 / −1,4 / −1,6 |

   - De K = 2 à K = 20, OBRS a un meilleur FID dans **15 cas sur 15**.
   - À K = 1,5 les deux méthodes sont égales. À K = 50, DRS est légèrement meilleur (−1,4) : les deux approchent le rejet exact, et la différence se joue sur quelques échantillons.
   - Précision et rappel restent indiscernables.
   - C'est cohérent avec la slide 66 du prof, où OBRS domine DRS sur la plage utile des budgets.
4. **La truncation échoue sur ce G (H3).**
   - La précision monte (0,97 à ψ = 0,5), mais le rappel s'effondre (0,35 à ψ = 1).
   - Le FID explose, et la répartition des classes empire (« 2 » à 2 %, « 9 » à 26 %).
   - Conforme à BigGAN : sans régularisation orthogonale à l'entraînement, G ne supporte pas la truncation.
   - C'est un « informative failure » à montrer : changer la loi de z n'est **pas** équivalent à corriger avec D.
5. **Coût :** K = 10 coûte environ 6 s sur CPU pour 10 000 images (contre 0,2 s), et beaucoup moins sur GPU. Le gain marginal devient faible au-delà de K ≈ 10.

## 5. Points à expliquer honnêtement (slide 86)

- **Estimateur, pas courbe théorique.** La précision et le rappel k-NN estiment des points support à partir d'un échantillon fini. La Prop. 3.3 porte sur la courbe PRD théorique (α_λ, β_λ). On teste la **tendance prédite**, pas l'égalité $\alpha' = K\alpha$.
- **D n'est pas optimal.** Il est entraîné contre un G qui bouge, et il a un peu mémorisé ses images d'entraînement (AUC entre images d'entraînement et MNIST test ≈ 0,58). Son logit n'est donc qu'une estimation de $\log p/\hat p$. Piste : recalibrer D sur `val` avec G figé (DRS et MH-GAN le font).
- **Features maison.** Le FID utilise un CNN MNIST, pas Inception : seules les comparaisons internes ont un sens.
- **Coût.** OBRS coûte K fois plus de passes de G et D. Il faut le dire et le chiffrer (`seconds`, `n_proposals`).

## 6. Slides (squelette : `slides/inference.tex`)

| # | Slide | Contenu |
|---|---|---|
| 1 | Question | G figé, baseline (P/R, déficit de « 2 »), question en une ligne |
| 2 | Méthode | $a_{\rm OBRS}$ dans la notation du prof, Prop. 3.3 → prédiction, témoins, coût |
| 3 | Résultats | `figures/test/pr_plane.pdf` et `figures/test/fid_budget.pdf` |
| 4 | Analyse | `figures/test/classes.pdf`, seuil dur contre OBRS (échec instructif), limites |

## 7. Suite

1. Entraîner `baseline_v1` sur Juliet (`scripts/train.sh`), puis lancer `scripts/sweep.sh runs/baseline_v1 best val` puis `... test`.
2. Choisir K sur `val`, puis écrire `checkpoints/sampling.yaml` (par exemple `{method: obrs, K: 10}`) une fois le groupe d'accord.
3. Extension si le temps le permet :
   - recalibrer D avec G figé, sur `val` ;
   - ou DGflow + OBRS (raffiner puis rejeter).
4. Combiner avec les meilleurs runs des volets Données et Entraînement : `evaluate.py --run runs/<leur_run>`.
