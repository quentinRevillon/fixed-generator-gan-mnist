# Rapport (sources LaTeX)

Le PDF rendu est `report.pdf`, à la racine du dépôt : page de garde + 5 pages de contenu (consigne de l'énoncé).

| Fichier | Contenu |
|---|---|
| `report.tex` | document principal : page de garde, résumé, inclusion des parties |
| `preamble.tex` | mise en page (deux colonnes, polices, couleurs, encadré `\takeaway{...}`) |
| `sections/*.tex` | une partie par fichier : `intro`, `fid`, `data`, `training`, `inference`, `combination` |
| `references.tex` | bibliographie (`\cite{clé}`) |
| `figures/*.pdf` | figures incluses (tracées par les scripts `figscripts/` de la branche `quentin/comparaison`) |

Compiler, depuis ce dossier (pdflatex, deux passes) puis remplacer le PDF de la racine :

```bash
python tools/build.py          # -> report/report.pdf ; affiche erreurs, débordements, pages
cp report.pdf ../report.pdf
```

Vérifier après chaque modification que le PDF fait toujours **6 pages** (garde + 5).
