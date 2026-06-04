"""
poule_strategy.py
-----------------
Vindt de voorspelling met de hoogste VERWACHTE PUNTEN gegeven jouw poule-regels.

Het Poisson-model geeft een kansverdeling over alle mogelijke uitslagen.
De 'meest waarschijnlijke uitslag' (de modus) is alleen optimaal als de poule
ALLEEN punten geeft voor een exact correcte gok. Zodra er deelpunten zijn voor
'winnaar goed' of 'doelsaldo goed', is de optimale gok soms een andere uitslag —
typisch iets dat de winnaar-kans maximaliseert.

Dit is een rechttoe rechtaan optimalisatie: probeer elke voorspelling (h, a),
bereken de verwachte score gegeven de kansmatrix, kies de hoogste.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ScoringRules:
    """Puntentelling per wedstrijd.

    cumulative=True : elk criterium telt onafhankelijk mee.
      bv. precies correct met 2-1 = exact + doelsaldo + winnaar = 10+6+4 = 20
    cumulative=False: alleen de hoogste haakje telt.
      bv. precies correct met 2-1 = 10 (de andere zitten erin verwerkt)
    """
    exact: int = 10
    goal_diff: int = 6
    winner: int = 4
    cumulative: bool = True


def points(predict_h: int, predict_a: int, actual_h: int, actual_a: int,
           rules: ScoringRules) -> int:
    """Punten voor één voorspelling tegen één werkelijke uitslag."""
    exact = (predict_h == actual_h and predict_a == actual_a)
    goal_diff_match = (predict_h - predict_a) == (actual_h - actual_a)
    sgn_pred = (predict_h > predict_a) - (predict_h < predict_a)
    sgn_act = (actual_h > actual_a) - (actual_h < actual_a)
    winner_match = sgn_pred == sgn_act

    if rules.cumulative:
        p = 0
        if winner_match: p += rules.winner
        if goal_diff_match: p += rules.goal_diff
        if exact: p += rules.exact
        return p
    # niet-cumulatief: alleen het beste criterium
    if exact: return rules.exact
    if goal_diff_match: return rules.goal_diff
    if winner_match: return rules.winner
    return 0


def expected_points(predict_h: int, predict_a: int, matrix: np.ndarray,
                    rules: ScoringRules) -> float:
    """Verwachte punten voor één voorspelling, gegeven de hele kansverdeling."""
    n = matrix.shape[0]
    e = 0.0
    for i in range(n):
        for j in range(n):
            e += matrix[i, j] * points(predict_h, predict_a, i, j, rules)
    return e


def optimal_prediction(matrix: np.ndarray, rules: ScoringRules,
                       max_search: int = 6) -> tuple[int, int, float]:
    """
    Doorzoek alle uitslagen tot max_search-max_search en kies degene met de
    hoogste verwachte punten. Returns: (home_goals, away_goals, expected_points).
    """
    best_h, best_a, best_ev = 0, 0, -1.0
    for h in range(max_search + 1):
        for a in range(max_search + 1):
            ev = expected_points(h, a, matrix, rules)
            if ev > best_ev:
                best_h, best_a, best_ev = h, a, ev
    return best_h, best_a, best_ev


def top_n_predictions(matrix: np.ndarray, rules: ScoringRules,
                      n: int = 5, max_search: int = 6) -> list[tuple[int, int, float]]:
    """Top-n voorspellingen op basis van verwachte punten — handig om alternatieven te zien."""
    results = []
    for h in range(max_search + 1):
        for a in range(max_search + 1):
            ev = expected_points(h, a, matrix, rules)
            results.append((h, a, ev))
    return sorted(results, key=lambda x: -x[2])[:n]


if __name__ == "__main__":
    # snelle demo van de scorefunctie
    rules = ScoringRules()  # 4/6/10 cumulatief
    cases = [
        (2, 1, 2, 1, "exact correct"),
        (2, 1, 3, 2, "winnaar + doelsaldo, niet exact"),
        (2, 1, 2, 0, "winnaar goed, doelsaldo fout"),
        (2, 1, 0, 2, "alles fout"),
        (1, 1, 2, 2, "voorspelt gelijk, wordt ander gelijk"),
    ]
    for ph, pa, ah, aa, note in cases:
        p = points(ph, pa, ah, aa, rules)
        print(f"  voorspelt {ph}-{pa}, werd {ah}-{aa}  -> {p} pt  ({note})")
