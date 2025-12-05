import numpy as np
from pymcdm.methods import TOPSIS, PROMETHEE_II
from pymcdm.helpers import rrankdata

METHOD = "SAW"   # "TOPSIS", "PROMETHEE", "SAW"

alternatives = [
    "Scénario 1 : H2 100% Vendée",
    "Scénario 2 : H2 Vendée + import régional",
    "Scénario 3 : Import H2 national/UE",
]

decision_matrix = np.array([
    [6.0, 1.0, 120, 4.5],
    [5.0, 1.5,  80, 4.0],
    [4.0, 3.0,  30, 3.0],
], dtype=float)

weights = np.array([0.30, 0.35, 0.20, 0.15], dtype=float)
types   = np.array([-1, -1,  1,  1], dtype=int)

def run_topsis(M, w, t):
    model = TOPSIS()
    prefs = model(M, w, t)
    ranks = rrankdata(prefs)
    return prefs, ranks

def run_promethee(M, w, t, pref_func="usual"):
    model = PROMETHEE_II(pref_func)
    prefs = model(M, w, t)
    ranks = (-prefs).argsort().astype(int) + 1
    return prefs, ranks

def run_saw(M, w, t):
    M = M.astype(float).copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        if t[j] == 1:  # bénéfice
            M[:, j] = (col - col.min()) / (col.max() - col.min())
        else:          # coût
            M[:, j] = (col.max() - col) / (col.max() - col.min())
    prefs = M @ w
    ranks = (-prefs).argsort().astype(int) + 1
    return prefs, ranks

if METHOD == "TOPSIS":
    preferences, ranking = run_topsis(decision_matrix, weights, types)
elif METHOD == "PROMETHEE":
    preferences, ranking = run_promethee(decision_matrix, weights, types)
elif METHOD == "SAW":   # utilité multi‑critère additive
    preferences, ranking = run_saw(decision_matrix, weights, types)
else:
    raise ValueError("Choisir METHOD parmi 'TOPSIS', 'PROMETHEE', 'SAW'.")

print(f"\n🔎 Résultats MCDA ({METHOD}) – Scénarios H2 :")
for alt, pref, rank in zip(alternatives, preferences, ranking):
    print(f"  Rang {int(rank)} | Score = {pref:.3f} | {alt}")
