"""Compile le rapport (pdflatex deux fois) et affiche pages, erreurs, débordements et références indéfinies.

Usage, depuis rapport/report/ :
    python tools/build.py            # report.tex -> report.pdf
    python tools/build.py fid        # une seule partie : build/fid/part.pdf (préambule + sections/fid.tex)
"""
import os
import re
import subprocess
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def part_source(part):
    """Source LaTeX minimal qui ne contient qu'une partie, pour la compiler seule."""
    return ('\\documentclass[10pt,twocolumn]{article}\n\\input{preamble}\n\\begin{document}\n'
            f'\\input{{sections/{part}}}\n{{\\footnotesize\\input{{references}}}}\n\\end{{document}}\n')


def pdflatex(tex_name, outdir):
    """Deux passes de pdflatex ; renvoie (code de retour, log de la dernière passe)."""
    os.makedirs(outdir, exist_ok=True)
    cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={outdir}', tex_name]
    for _ in range(2):
        run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='latin-1')
    return run.returncode, run.stdout


def main(part=None):
    """Compile le rapport entier ou une partie, et résume le résultat."""
    if part:
        outdir = os.path.join(ROOT, 'build', part)
        os.makedirs(outdir, exist_ok=True)
        tex = f'build/{part}/part.tex'
        with open(os.path.join(ROOT, tex), 'w', encoding='utf-8') as f:
            f.write(part_source(part))
        code, log = pdflatex(tex, outdir)
        pdf = os.path.join(outdir, 'part.pdf')
    else:
        code, log = pdflatex('report.tex', ROOT)
        pdf = os.path.join(ROOT, 'report.pdf')
    errors = [line for line in log.splitlines() if line.startswith('!')]
    overfull = [float(x) for x in re.findall(r'Overfull \\[hv]box \(([\d.]+)pt too', log)]
    undefined = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)))
    print('exit', code, '| erreurs', errors[:5])
    print('overfull > 1 pt :', [x for x in overfull if x > 1][:10])
    print('références indéfinies :', undefined)
    if os.path.exists(pdf):
        print('pages :', len(pymupdf.open(pdf)), '->', pdf)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
