"""Rend chaque page de report.pdf en PNG (110 dpi) dans un dossier neuf : <dossier>/p-01.png, p-02.png, ...

Usage, depuis rapport/report/ :  python tools/render.py rounds/r1
Le dossier ne doit pas exister : on ne réécrit jamais un dossier qu'un relecteur peut être en train de lire.
"""
import os
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    out = os.path.join(ROOT, sys.argv[1])
    os.makedirs(out, exist_ok=False)
    doc = pymupdf.open(os.path.join(ROOT, 'report.pdf'))
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=110).save(os.path.join(out, f'p-{i:02d}.png'))
    print(len(doc), 'pages ->', out)
