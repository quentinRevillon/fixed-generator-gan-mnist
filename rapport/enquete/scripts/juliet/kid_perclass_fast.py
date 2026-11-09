"""Même mesure que kid_perclass.py, avec un FID calculé par valeurs propres : Tr((S_r S_g)^(1/2)) = somme des racines
des valeurs propres de S_r S_g. Identique à scipy.linalg.sqrtm à 1e-4 près, beaucoup plus rapide en 2 048 dimensions.

Usage, depuis la racine du repo (branche noe/training) :
    PYTHONPATH=. python scripts/kid_perclass_fast.py <manifeste.csv> <sortie.csv>
"""
import sys

import numpy as np

import scripts.kid_perclass as kp


def fid_eig(f_real, f_fake):
    """FID = ||mu_r - mu_g||^2 + Tr(S_r) + Tr(S_g) - 2 somme(sqrt(valeurs propres de S_r S_g))."""
    f1, f2 = f_real.double().numpy(), f_fake.double().numpy()
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    eig = np.linalg.eigvals(s1 @ s2).real.clip(min=0)
    return float(((f1.mean(0) - f2.mean(0)) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(eig).sum())


if __name__ == '__main__':
    kp.fid = fid_eig
    kp.main(sys.argv[1], sys.argv[2])
