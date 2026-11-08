# Rapport final : enquête, expériences et plan

> État au 8/11/2026. Rapport `report.pdf` le **11/11 à 23h59** (5 pages + page de garde, annexes non lues). Code le **12/11 à 05h00**.
> Détail de chaque volet, sources ligne par ligne : [`enquete/`](enquete/).

---

## 0. Ce qu'il faut retenir

| Question | Réponse courte |
|---|---|
| Des collaborateurs ont-ils poussé du nouveau ? | **Non.** Dernier commit : 4/11 22h15 (modèle final sur `main`). |
| Les chiffres des slides sont-ils justes ? | **Oui, à l'arrondi près**, sauf « Noise 0.3 : 0 of 8 » : 3 runs observés, 5 comptés sains d'office. |
| Le DRS fait-il baisser les performances ? (prof) | **En partie.** Il aide la baseline, ne change rien à RpGAN en FID Inception, et **dégrade RpGAN en FID CNN** (9,9 → 14,2, 3 seeds sur 3). |
| Pourquoi ? | Le D de RpGAN ne sépare plus vrai et faux (AUC 0,57 contre 0,75) : le tri est presque aléatoire et déforme le mélange des chiffres, que le FID CNN voit fortement. |
| Peut-on réparer ? | **Oui : en recalibrant D** (l'étape d'Azadi que nous avions sautée), le tri améliore le modèle final en FID Inception : **22,0 → 20,2 (DRS), 19,7 (OBRS)**, sur les 3 seeds, rappel inchangé (Juliet, §5.4). |
| Faut-il garder le FID CNN ? (prof) | **Oui, mais comme diagnostic, pas comme mesure principale.** Il mesure surtout le mélange des chiffres. La mesure principale devient le FID Inception (celle de la plateforme). |
| « The loss gives the largest gain » ? | **Faux pour le FID Inception.** La baseline commune n'a pas d'EMA. Sans EMA, RpGAN + R1 = baseline en Inception (37,7 contre 38,7, p = 0,34) ; c'est l'EMA qui donne 37,7 → 22,1 (p = 0,0001). La perte + R1 corrige le mélange des chiffres (FID CNN 182 → 20). Vérifié sur 3 seeds (§2). |
| Bloquant pour le rendu | `main` ne contient pas le code qui entraîne le modèle rendu (RpGAN + EMA est seulement sur `noe/training`). **À fusionner avant le 12/11.** |

**Notions utilisées**
- **FID Inception** : distance entre vraies et fausses images, dans les features d'Inception (réseau ImageNet). C'est la mesure de la plateforme.
- **FID CNN** : même formule, dans les features de notre petit classifieur de chiffres.
- **G brut / G_EMA** : poids de G à la dernière étape / moyenne mobile de ces poids (c'est G_EMA qu'on rend).
- **AUC de D** : probabilité que D note une vraie image plus haut qu'une fausse. 0,5 = D ne distingue rien.
- **p** : probabilité d'un écart aussi grand par le seul hasard des seeds. p < 0,05 = écart réel.

---

## 1. Expériences relancées (CPU local, faites le 8/11)

Toutes sur MNIST test, 10 000 images sauf mention. Scripts et CSV : `enquete/scripts/`, `enquete/results/`.

| # | Volet | Expérience | Résultat clé |
|---|---|---|---|
| L1 | FID | Dégradations dosées de vraies images, vues par les deux FID | 1 chiffre sur 10 retiré : **Inception +2 %, CNN +181 %**. Bruit sur 25 % des images : **Inception +285 %**, CNN +108 % (dans son propre bruit). |
| L2 | FID | Part de la variance des features due au chiffre | **CNN 67 %**, Inception 26 % |
| L3 | FID | Baseline remise dans les bonnes proportions de chiffres | FID CNN **162 → 32** (−82 %), FID Inception +16 % : le FID CNN de la baseline est surtout son mélange de chiffres |
| L4 | FID | Accord entre les deux FID (Spearman) | Baseline après l'époque 100 : **ρ = −0,885** (ordre inverse) ; entre variantes RpGAN : 0,43 (n.s.) |
| L5 | FID | Biais selon le nombre d'images, plancher réel-réel | CNN : 62,7 (N = 250) → 2,9 (N = 10 000) |
| L6 | DRS | Notes de D, AUC, acceptation par chiffre, 5 modèles | AUC baseline 0,75-0,76 ; **RpGAN 0,55-0,56** ; 43 % gardées contre 30 % |
| L7 | DRS | DRS avec D recalibré (couche logistique, comme Azadi), CNN seulement | Baseline : FID CNN 174 → 111 ; RpGAN : neutre en CNN. **Confirmé et complété en Inception sur Juliet (G1, §5.4)** |
| L8 | DRS | DRS sur le G brut avec son D | Dégrade aussi (15,0 → 17,0) : l'EMA n'est pas la seule cause |
| L9 | Training | G_EMA contre G brut (= run sans EMA), 18 runs | FID CNN commun : baseline 182 → RpGAN + R1 brut **19,7** → + EMA **9,9** |
| L10 | Training | FID Inception G brut, **3 seeds, 10 000 images** (contrôle : redonne le 21,15 du groupe à 1e-6 près) | Sans EMA : **37,7 ± 1,0** (baseline 38,7) ; avec EMA : 22,1. Voir §2 |
| L11 | Données | Tests statistiques sur tous les runs | Bruit 0,3 : p = 0,46 (n.s.) ; effondrements 0/3 contre 0/3 de la référence sur les mêmes seeds |
| L12 | Combinaison | Audit de chaque chiffre des slides | Tout est juste sauf « 0 of 8 » |

---

## 2. D'où vient le gain du modèle final ? (protocole commun, 3 seeds, test)

| Modèle | EMA | FID Inception | FID CNN |
|---|---|---|---|
| Baseline (BCE) | non | 38,7 ± 1,2 | 182 ± 19 |
| RpGAN + R1 (G brut) | non | **37,7 ± 1,0** | 19,7 ± 4,2 |
| RpGAN + R1 + EMA (rendu) | oui | 22,1 ± 1,3 | 9,9 ± 2,5 |
| Baseline + EMA | oui | *en cours (G2, job 145826)* | *en cours* |

| Effet (Welch, 3 seeds contre 3) | FID Inception | FID CNN |
|---|---|---|
| Perte + R1 (baseline → RpGAN brut) | −1,0 (p = 0,34, **rien**) | −162 (p = 0,003) |
| EMA (RpGAN brut → RpGAN + EMA) | **−15,5 (p = 0,0001)** | −9,8 (p = 0,03) |

- **Perte + R1** : corrige le **mélange des chiffres** (écart-type des proportions 3,3 → 1,4 point) et supprime les effondrements. Le FID CNN le voit ; Inception non.
- **EMA** : améliore l'**aspect des images**. C'est elle qui fait tout le gain en FID Inception, donc sur la plateforme.
- **Nouvelle conclusion** : *« Perte + R1 : bons chiffres et stabilité. EMA : images plus propres. Les deux sont nécessaires. »*
- **Question ouverte, la plus importante** : l'EMA seule suffirait-elle sur la baseline ? À 100 époques, elle ne faisait que −10 % de FID CNN sur la BCE, mais son effet Inception n'a jamais été mesuré → **G2 devient la priorité n° 1**.
- **Réplication indépendante** retrouvée sur Juliet (jobs 143380-143381 du 4/11, jamais rapatriés ; CUDA, meilleur checkpoint, sans EMA) : baseline **36,4 ± 1,0**, RpGAN + R1 **36,1 ± 1,2**, R3GAN 37,7 ± 0,9 (FID Inception, 3 seeds). Même conclusion : sans EMA, la perte ne change rien en Inception. Bruit 0,1 et sélection 50 % : 1 et 2 seeds sur 3 effondrées.
- Scripts et CSV : `enquete/scripts/ema/`, `enquete/results/ema/`, `enquete/results/juliet/`.

---

## 3. Expériences sur Juliet

Lancées le 8/11 depuis `~/gan-a2-noe` (worktree de **`noe/training`**, la seule branche avec l'EMA ; RpGAN + bruit plante sur `quentin/comparaison`). Script : `scripts/juliet2.sh`, configs `configs/juliet2/`. Commandes des autres : `enquete/<volet>.md`, section 6.

| Prio | # | Expérience | Question tranchée | État |
|---|---|---|---|---|
| **P0** | G0 | Rapatrier les 5 runs `noise_03_e100` restés sur Juliet | « 0 of 8 » vrai ou faux | **non poussés** sur `nico/preprocessing` (seulement `noise_03_s0-2`) : à demander à Nicodème |
| **P1** | G1 | DRS avec **D recalibré** + OBRS, mesuré en **Inception**, baseline et RpGAN, 3 seeds × 3 tirages | La recalibration rend-elle le DRS utile au sens de la plateforme ? | **fait** : oui pour RpGAN (22,0 → 19,7), non pour la baseline (§5.4). `enquete/results/juliet/drs_recalib_test.csv` |
| **P0** | G2 | **Baseline + EMA** au protocole commun, 3 seeds | L'EMA seule suffit-elle ? (dernière case du tableau perte × EMA) | **soumis** (job 145826, seeds 0-2 ; G_raw = baseline sans EMA, même run) |
| **P1** | G3 | Runs de sélection (`sel_95/90/75/50`) mesurés en **Inception + VGG** | Casser la circularité (sélection et FID CNN utilisent le même CNN) | **soumis** (job 145834) : les 52 `G.pth` poussés par Nicodème (dernier checkpoint, 120k mises à jour), en Inception + VGG + CNN |
| **P1** | G4 | **BCE + bruit 0,3** au protocole commun, 3 à 8 seeds | Seul cas où le bruit peut agir (RpGAN ne s'effondre jamais) | **soumis** (job 145826, 5 seeds, avec EMA ; G_raw = sans EMA) |
| **P1** | G9 | **KID Inception** et **FID par chiffre** (Inception et CNN, 1 000 images par chiffre), 6 configs × 3 seeds + plancher réel | Avis indépendant : le KID n'a pas de biais lié à N ; le FID par chiffre sépare « bon mélange » et « bons traits » | **soumis** (job 145835) |
| **P1** | G10 | **Tests de sensibilité à 10 000 images** (Inception était à 500) : bruit, flou, chimères, effacement, épaississement, chiffres retirés, déséquilibres | Trancher la contradiction avec Lucic et al. 2018 (« FID heavily influenced by missing modes ») | **soumis** (job 145836) |
| P2 | G5 | BCE sur logits ± R1 (≈ 10 lignes dans `utils.py`) | Séparer « relativiste » et « R1 » | à faire (code à ajouter) |
| P2 | G6 | Contrôle **50 % au hasard** (celui de DeVries) | Moins d'images, ou images choisies ? | à faire (patch de 2 lignes) |
| P2 | G7 | Temps total de génération et taux d'acceptation du DRS | Exigé par l'énoncé (p. 82), retiré des slides | **inclus dans G1** (colonne `seconds`) |
| P3 | G8 | γ en Inception, R3GAN + EMA au commun, EMA 0,9999, 400 époques | Approfondissements | si le temps le permet |

À ne pas relancer : `sel_50`, `sel_75` (résultat net, FID ≥ 216).

---

## 4. Structure du rapport (5 pages)

Gabarit de chaque méthode, 4 lignes : **Attendu (papier) → Observé (chiffre, p) → Pourquoi → Quand ça ne marche pas.**

| # | Partie | Message | Figure / tableau | Pages |
|---|---|---|---|---|
| 1 | **Protocole** | Split 50k/10k/10k, 10 000 images, test, 3 seeds, moyenne ± écart-type. Tableau des protocoles **corrigé** (mises à jour lues, EMA, checkpoint). | Tableau | 0,5 |
| 2 | **Évaluation (FID)** | Deux FID, deux rôles (§5.1) | Figure de sensibilité (`fid_sensitivity.png`) + histogramme des chiffres | 0,8 |
| 3 | **Données** | Sélection : précision ↑, rappel ↓, FID jamais meilleur. Bruit : aucun effet mesurable. | `data_results.png` + compte d'effondrements corrigé | 0,8 |
| 4 | **Entraînement** | RpGAN + R1 corrige le mélange des chiffres et ne s'effondre jamais ; **l'EMA fait tout le gain Inception** | Tableau perte × EMA (§2), en Inception et CNN + `cmp_loss_effect.png` | 0,9 |
| 5 | **Inference (DRS)** | Aide un G imparfait avec D tel quel ; sur un bon G, il faut recalibrer D (22,0 → 19,7) | Tableau baseline / RpGAN × sans DRS / DRS / DRS recalibré + acceptation + temps | 0,8 |
| 6 | **Combinaison** | Les effets ne s'additionnent pas ; seul perte + EMA dépasse le bruit | Tableau commun complet (Inception, P/R VGG, CNN, acceptation) | 0,7 |
| 7 | **Conclusion** | Modèle rendu : RpGAN + R1 + EMA, seed 1. Plateforme : **FID 21,59, P 0,702, R 0,585**. | — | 0,3 |

---

## 5. Contenu par partie

### 5.1 FID : la réponse au prof

| | FID Inception | FID CNN |
|---|---|---|
| Voit surtout | l'aspect des images (bruit, traits) | le **mélange des chiffres** |
| Chiffre manquant | presque aveugle (+2 %) | très sensible (+181 %) |
| Variation entre seeds | 2,6-4,5 % | 8,7-20 % |
| Comparable aux papiers / plateforme | oui | non |
| Rôle dans le rapport | **mesure principale** | **diagnostic**, toujours avec l'histogramme des chiffres |

- **Pour** le FID CNN : dans le domaine, pratique publiée (Bińkowski et al. 2018 remplacent Inception par un classifieur LeNet sur MNIST), voit la perte de modes, quasi gratuit.
- **Contre** : sature sur les bons modèles, non comparable, **circulaire** pour la sélection (même CNN), γ choisi avec lui.
- Le **÷18 contre ÷1,75** s'explique : RpGAN corrige surtout le mélange des chiffres. À chiffre fixé, son gain en FID CNN n'est que de 11 %.

### 5.2 Données

| Méthode | Attendu (papier) | Observé | Pourquoi l'écart |
|---|---|---|---|
| Sélection (DeVries 2020) | FID 21,4 → 12,6 à 50 % (ImageNet) | 95 % = référence ; 50 % : FID 76 → 331, rappel 0,72 → 0,38 | G non conditionnel, MNIST déjà propre, la capacité ne limite pas (le même G atteint 9,9) |
| Bruit (Sønderby 2017) | stabilité, aucun gain de FID annoncé | FID inchangé (p = 0,10-0,46) ; effondrements non démontrés | σ = 0,3 trop petit pour faire se recouvrir les distributions ; redondant avec R1 |
| Effondrement | — | 15/34 runs BCE | sigmoïde + `BCELoss` : le gradient de G s'annule (vérifié) |

### 5.3 Entraînement

| Élément | Attendu | Observé | Pourquoi |
|---|---|---|---|
| RpGAN (Jolicoeur-Martineau 2019) | gain fort si la baseline est instable | 0 effondrement sur 9 runs | gradient tant qu'un écart réel/faux existe |
| R1 (Mescheder 2018) | convergence locale | γ = 0,1 diverge, γ = 1 meilleur, γ = 10 plus lent | γ = 1 = γ critique de la perte logistique (hypothèse) |
| R1 sans R2 (R3GAN 2024) | **échoue** sur StackedMNIST | marche ; R2 n'apporte rien | tâche plus simple, petits réseaux (hypothèse) |
| EMA (Yazıcı 2019) | −11 à −29 % de FID | RpGAN : Inception −41 %, CNN −50 % ; BCE (100 ép.) : CNN −10 % | l'EMA moyenne des oscillations autour d'un équilibre, pas une dérive lente |

### 5.4 Inference (DRS)

Juliet (job 145828), protocole commun, 3 seeds × 2 tirages, contre test. Recalibrer = remplacer la dernière couche de D par une régression logistique entraînée sur val contre G (Azadi et al. 2019, §4.2).

| G | Tri | AUC de D | FID Inception | P / R VGG | FID CNN | Acceptation |
|---|---|---|---|---|---|---|
| Baseline | aucun | — | 38,7 | 0,47 / 0,50 | 183 | 100 % |
| Baseline | DRS, D tel quel | 0,75 | 34,3 | 0,51 / 0,50 | 154 | 30 % |
| Baseline | DRS, D recalibré | 0,83 | 36,3 | 0,46 / 0,51 | 103 | 32 % |
| Baseline | **OBRS, D tel quel** | 0,75 | **33,8** | 0,51 / 0,50 | 145 | 33 % |
| RpGAN + R1 + EMA | aucun | — | 22,0 | 0,59 / 0,59 | 9,9 | 100 % |
| RpGAN + R1 + EMA | DRS, D tel quel | 0,57 | 22,1 | 0,59 / 0,59 | 14,4 | 42 % |
| RpGAN + R1 + EMA | DRS, D recalibré | 0,78 | 20,2 | 0,62 / 0,59 | 10,0 | 32 % |
| RpGAN + R1 + EMA | **OBRS, D recalibré** | 0,78 | **19,7** | 0,62 / 0,59 | 10,2 | 33 % |

- **Quand D distingue déjà bien (baseline)** : le tri marche avec D tel quel. La recalibration corrige le mélange des chiffres (CNN 154 → 103) mais garde des images moins nettes (Inception moins bon).
- **Quand D ne distingue plus (RpGAN, AUC 0,57)** : D tel quel trie au hasard ; recalibré (AUC 0,78), il redevient utile : **−1,3 à −3,0 points d'Inception sur les 3 seeds**, précision +0,03, rappel inchangé.
- **Coût** : 10 000 images en 0,02 s sans tri, 0,08-0,14 s avec DRS, 0,16 s avec OBRS (GPU A100) ; le coût réel est le facteur ≈ 3 de propositions.
- Règle : **mesurer l'AUC de D avant de trier ; la recaler si elle est proche de 0,5.**

### 5.5 Combinaison

| Effet | Δ FID Inception | Verdict |
|---|---|---|
| Perte + R1 + EMA | −16,5 (p < 0,001) | réel, **dû à l'EMA** (perte + R1 seule : −1,0, p = 0,34) |
| Bruit 0,3 sur RpGAN | −0,7 (p = 0,46) | bruit des seeds |
| DRS sur baseline | −4,3 (3 seeds sur 3) | réel, compromis P ↑ / R ↓ |
| DRS sur RpGAN | −0,1 (Inception) ; +4,2 en CNN (p = 0,03) | inutile, voire nuisible, **sauf avec D recalibré** (−2,3 en moyenne, 3 seeds sur 3) |

Avec 3 seeds, un écart de moins de **≈ 3,6 points d'Inception** est indétectable.

---

## 6. Corrections à reporter (slides → rapport)

| Où | Actuel | Correction |
|---|---|---|
| Three levers | RpGAN + R1, Huang et al. 2024 | RpGAN (Jolicoeur-Martineau 2019) + R1 (Mescheder 2018) **+ EMA**, recette R3GAN sans R2 |
| Protocol | Data : 120k updates | lu à 60k ; ajouter colonnes « EMA » et « checkpoint » |
| Protocol | precision, recall | préciser : CNN (k = 3) ; plateforme : VGG (k = 5) |
| Data : collapsed | Noise 0.3 : 0 of 8 | **0 of 3, non significatif** (ou vrai compte après G0) |
| Training | RpGAN : none | RpGAN + R1 (γ ≥ 1) : 0 sur 9 ; sans R1 : diverge |
| Sampling : method | γ ≈ −6,4 ; D as trained | γ ≈ −6,4 **pour la baseline** ; D non recalibré (le papier recalibre) |
| Common protocol | The loss improves both FIDs | Perte + R1 + EMA ; **baseline sans EMA** |
| Conclusion | The loss gives the largest gain | Loss + R1: right digit mix, no collapse. EMA: the Inception gain (37.7 → 22.1). |
| Conclusion | Noise 0.3 : fewer collapses | aucun effet mesurable |
| Partout | γ pour R1 et pour le DRS | deux symboles différents |

---

## 7. Décisions pour le groupe

1. **Rendre le modèle avec OBRS + D recalibré ?** Gain attendu ≈ −2 points de FID sur la plateforme ; il faut que `generate.py` charge D, la couche recalibrée et fasse le tri. Décision du groupe avant le 12/11 05h00.
1. **Fusionner le code de `noe/training` dans `main`** avant le 12/11 05h00 (sinon le modèle rendu n'est pas reproductible).
2. Valider le changement de mesure principale (Inception) et le nouveau rôle du FID CNN.
3. G1, G2, G3, G4 tournent sur Juliet (compte de Quentin). G0 : demander à Nicodème de pousser les 5 runs `noise_03_e100`.
