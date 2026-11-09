"""Compte les mots visibles du rapport (texte extrait de report.pdf, page de garde et bibliographie exclues).

Usage, depuis rapport/report/ :  python tools/words.py
"""
import os
import re

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    doc = pymupdf.open(os.path.join(ROOT, 'report.pdf'))
    text = '\n'.join(page.get_text() for page in list(doc)[1:])
    text = text.split('References')[0]
    print(len(re.findall(r"[A-Za-zÀ-ÿ0-9][\w'.,%-]*", text)), 'mots visibles (hors garde et références)')
