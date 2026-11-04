# Partie Data : texte oral (environ 1 min 30)

## Data: papers (10 s)
- Deux leviers sur ce que voit le GAN, sans toucher au modèle.
- **Instance selection** (DeVries et al., NeurIPS 2020) : retirer les images atypiques *avant* l'entraînement.
- **Instance noise** (Sønderby et al., ICLR 2017) : bruiter les entrées de D pendant l'entraînement.

## Data: selection (25 s)
- On score chaque image avec une **gaussienne par chiffre** dans les **features** du CNN : $\mu_c$ et $\Sigma_c$ sont la moyenne et la covariance des features du chiffre $c$.
- $s(x)$ élevé = image **typique**. On garde 95, 90 ou 50 % de chaque chiffre.
- À gauche les plus typiques, à droite les moins typiques : des **styles rares** (7 barré, 4 fermé), **pas des erreurs**.
- **Différence avec le papier :** sur ImageNet, on retire de vrais déchets ; MNIST est déjà propre, donc on retire de la **diversité**.

## Data: noise (10 s)
- On ajoute $\sigma_t\,\varepsilon$ aux vraies et aux fausses images avant D, avec σ qui décroît de 0,1 ou 0,3 vers 0 en 100 époques (**annealing**).
- L'image : ce que voit D au fil des époques, de plus en plus net.

## Data: results (20 s)
- **Sélection :** la **precision** monte, le **recall** baisse, comme dans le papier. Mais le **FID** empire, alors que le papier l'améliore (de 17,2 à 9,1).
- Même à 95 %, on ne fait qu'égaler la référence : aucune rétention ne la bat.
- **Pourquoi :** le papier utilise un **generator conditionnel** (BigGAN, SAGAN) ; notre MLP ne l'est pas, et la sélection lui fait perdre des chiffres entiers.
- **Bruit :** FID et recall identiques à la référence.

## Data: collapsed runs (25 s)
- **15 runs sur 34 s'effondrent** sans bruit 0,3 dès les premières époques, quelle que soit la méthode.
- À droite : même config, seule la **seed** change ; le run effondré sort la même tache pour tout z.
- **Mécanisme :** D gagne tout de suite, sa **sigmoid sature**, et G reçoit un **gradient nul** (la `BCELoss` le perd numériquement).
- Même taux **avec ou sans sélection**.
- **Avec σ = 0,3, aucun run ne s'effondre** (0 sur 8) : le bruit empêche D de gagner trop vite.
- C'est exactement la motivation de Sønderby et Arjovsky & Bottou 2017 : élargir les supports pour qu'ils se **recouvrent**. Sønderby ne l'avait **pas testé en génération** ; chez nous, le bruit stabilise l'entraînement, mais n'améliore pas le FID.
