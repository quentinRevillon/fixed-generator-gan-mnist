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
| Pourquoi ? | Le D de RpGAN ne sépare plus vrai et faux (AUC 0,55 contre 0,75) : le tri est presque aléatoire et déforme le mélange des chiffres, que le FID CNN voit fortement. |
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
| L7 | DRS | DRS avec D recalibré (couche logistique, comme Azadi) | Baseline : FID CNN 174 → **111** (DRS actuel : 159) ; **87** avec OBRS. RpGAN : redevient neutre |
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
| Baseline + EMA | oui | *jamais mesuré (G2)* | *jamais mesuré* |

| Effet (Welch, 3 seeds contre 3) | FID Inception | FID CNN |
|---|---|---|
| Perte + R1 (baseline → RpGAN brut) | −1,0 (p = 0,34, **rien**) | −162 (p = 0,003) |
| EMA (RpGAN brut → RpGAN + EMA) | **−15,5 (p = 0,0001)** | −9,8 (p = 0,03) |

- **Perte + R1** : corrige le **mélange des chiffres** (écart-type des proportions 3,3 → 1,4 point) et supprime les effondrements. Le FID CNN le voit ; Inception non.
- **EMA** : améliore l'**aspect des images**. C'est elle qui fait tout le gain en FID Inception, donc sur la plateforme.
- **Nouvelle conclusion** : *« Perte + R1 : bons chiffres et stabilité. EMA : images plus propres. Les deux sont nécessaires. »*
- **Question ouverte, la plus importante** : l'EMA seule suffirait-elle sur la baseline ? À 100 époques, elle ne faisait que −10 % de FID CNN sur la BCE, mais son effet Inception n'a jamais été mesuré → **G2 devient la priorité n° 1**.
- Scripts et CSV : `enquete/scripts/ema/`, `enquete/results/ema/`.

---

## 3. Expériences à lancer sur Juliet

Je n'ai pas accès à Juliet depuis cette machine (clé protégée par passphrase). Partir de **`noe/training`** : c'est la seule branche avec l'EMA, et RpGAN + bruit plante sur `quentin/comparaison`. Commandes exactes : `enquete/<volet>.md`, section 6.

| Prio | # | Expérience | Question tranchée | Coût GPU |
|---|---|---|---|---|
| **P0** | G0 | Rapatrier les 5 runs `noise_03_e100` restés sur Juliet | « 0 of 8 » vrai ou faux | 0 |
| **P1** | G1 | DRS avec **D recalibré** + OBRS, mesuré en **Inception**, baseline et RpGAN, 3 seeds × 3 tirages | La recalibration rend-elle le DRS utile au sens de la plateforme ? | ≈ 40 min |
| **P0** | G2 | **Baseline + EMA** au protocole commun, 3 seeds | L'EMA seule suffit-elle ? (dernière case du tableau perte × EMA) | 3 runs ≈ 1 h en parallèle |
| **P1** | G3 | Runs de sélection (`sel_95/90/75/50`) mesurés en **Inception + VGG** | Casser la circularité (sélection et FID CNN utilisent le même CNN) | ≈ 20 min |
| **P1** | G4 | **BCE + bruit 0,3** au protocole commun, 3 à 8 seeds | Seul cas où le bruit peut agir (RpGAN ne s'effondre jamais) | 3-8 runs |
| P2 | G5 | BCE sur logits ± R1 (≈ 10 lignes dans `utils.py`) | Séparer « relativiste » et « R1 » | 6 runs |
| P2 | G6 | Contrôle **50 % au hasard** (celui de DeVries) | Moins d'images, ou images choisies ? | 3 runs |
| P2 | G7 | Temps total de génération et taux d'acceptation du DRS | Exigé par l'énoncé (p. 82), retiré des slides | minutes |
| P3 | G8 | γ en Inception, R3GAN + EMA au commun, EMA 0,9999, 400 époques | Approfondissements | plusieurs h |

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
| 5 | **Inference (DRS)** | Aide un G imparfait, inutile sur un bon G ; la recalibration de D répare | Tableau baseline / RpGAN × sans DRS / DRS / DRS recalibré + acceptation + temps | 0,8 |
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

| Cas | Effet | Mécanisme |
|---|---|---|
| Baseline, D tel quel | FID Inception 38,7 → 34,4 ; P +0,04 ; R −0,01 | D distingue bien (AUC 0,76) |
| Baseline, D recalibré | FID CNN 174 → 111 (à confirmer en Inception, G1) | AUC 0,75 → 0,83-0,85 |
| RpGAN, D tel quel | Inception = ; **CNN +4,2** | AUC 0,55 : tri presque au hasard qui déforme le mélange des chiffres |
| RpGAN, D recalibré | neutre | il reste peu à corriger sur un bon G |

Règle : **mesurer l'AUC de D avant d'utiliser le DRS.**

### 5.5 Combinaison

| Effet | Δ FID Inception | Verdict |
|---|---|---|
| Perte + R1 + EMA | −16,5 (p < 0,001) | réel, **dû à l'EMA** (perte + R1 seule : −1,0, p = 0,34) |
| Bruit 0,3 sur RpGAN | −0,7 (p = 0,46) | bruit des seeds |
| DRS sur baseline | −4,3 (3 seeds sur 3) | réel, compromis P ↑ / R ↓ |
| DRS sur RpGAN | −0,1 (Inception) ; +4,2 en CNN (p = 0,03) | inutile, voire nuisible |

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

1. **Fusionner le code de `noe/training` dans `main`** avant le 12/11 05h00 (sinon le modèle rendu n'est pas reproductible).
2. Valider le changement de mesure principale (Inception) et le nouveau rôle du FID CNN.
3. Qui lance G0-G4 sur Juliet, et quand (avant le 10/11 au soir pour écrire le 11/11).
