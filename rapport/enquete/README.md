# Enquête du 8/11/2026

Analyses détaillées derrière [`../PLAN.md`](../PLAN.md). Chaque fichier suit le même plan : attendu (papiers), observé, divergences, ce qui marche ou non, corrections des slides, expériences, sources.

| Fichier | Volet |
|---|---|
| `fid.md` | Justification et critique du FID CNN face au FID Inception |
| `data.md` | Sélection d'instances et bruit d'instance |
| `training.md` | RpGAN + R1 + EMA, ablation de l'EMA |
| `inference.md` | DRS : pourquoi il n'aide pas RpGAN, recalibration de D |
| `combinaison.md` | Audit des chiffres des slides, protocoles, cahier des charges du rapport |

`scripts/` : scripts des mesures locales (CPU). Ils importent `evaluate.py`, `sampling.py`, `model.py` du dépôt et prennent en argument un dossier de travail (MNIST, caches). Les chemins `scratchpad/...` cités dans les `.md` désignent ce dossier de travail local, non versionné.
`results/` : sorties chiffrées (CSV, figures) de ces scripts.
