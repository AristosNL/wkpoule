"""
model.py
--------
Zet twee Elo-ratings om naar een volledige kansverdeling over uitslagen.

Pijplijn per wedstrijd:
  1. Elo-verschil  -> verwachte doelsaldo (supremacy)  [gekalibreerd op de historie]
  2. supremacy + gemiddeld totaal  -> lambda_thuis, lambda_uit
  3. Poisson per team  -> matrix P(score_thuis, score_uit)
  4. (optioneel) Dixon-Coles-correctie op de vier laagste cellen
  5. matrix  -> P(thuiswinst), P(gelijk), P(uitwinst), en meest waarschijnlijke uitslag
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson

from elo import expected_score, HOME_ADVANTAGE


@dataclass
class Calibration:
    """Parameters die we uit de historie schatten."""
    goals_per_elo: float   # 'a': verwacht doelsaldo per Elo-punt verschil
    base_total: float      # gemiddeld aantal doelpunten per interland
    rho: float = -0.08     # Dixon-Coles lage-score-correctie (0 = uit)


def calibrate(history: list[dict], since_year: int = 2010) -> Calibration:
    """
    Schat 'a' (doelsaldo per Elo-punt) en het gemiddelde totaal uit de historie.
    Lineaire regressie door de oorsprong: supremacy = a * dr.
    """
    dr = np.array([h["dr"] for h in history if h["date"].year >= since_year])
    sup = np.array([h["supremacy"] for h in history if h["date"].year >= since_year])
    tot = np.array([h["total_goals"] for h in history if h["date"].year >= since_year])
    a = float(np.sum(dr * sup) / np.sum(dr * dr))
    return Calibration(goals_per_elo=a, base_total=float(tot.mean()))


def expected_goals(rating_home: float, rating_away: float, cal: Calibration,
                   neutral: bool = True) -> tuple[float, float]:
    """Verwachte goals (lambda) voor beide teams op basis van Elo."""
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    dr = (rating_home + home_adv) - rating_away
    supremacy = cal.goals_per_elo * dr
    lam_home = max(0.15, (cal.base_total + supremacy) / 2.0)
    lam_away = max(0.15, (cal.base_total - supremacy) / 2.0)
    return lam_home, lam_away


def _dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles-correctiefactor; raakt alleen 0-0, 1-0, 0-1, 1-1 aan."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_home: float, lam_away: float, cal: Calibration,
                 max_goals: int = 10) -> np.ndarray:
    """Matrix M[i, j] = kans op exact i goals thuis en j goals uit."""
    ph = poisson.pmf(np.arange(max_goals + 1), lam_home)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_away)
    m = np.outer(ph, pa)
    if cal.rho != 0.0:
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= _dc_tau(i, j, lam_home, lam_away, cal.rho)
    return m / m.sum()


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """(P_thuiswinst, P_gelijk, P_uitwinst) uit de scorematrix."""
    p_home = float(np.tril(matrix, -1).sum())   # onder de diagonaal
    p_draw = float(np.trace(matrix))            # de diagonaal
    p_away = float(np.triu(matrix, 1).sum())    # boven de diagonaal
    return p_home, p_draw, p_away


def most_likely_score(matrix: np.ndarray) -> tuple[int, int, float]:
    """Meest waarschijnlijke exacte uitslag (handig voor poules met exacte-score-punten)."""
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(i), int(j), float(matrix[i, j])


def rescale_matrix_to_outcome(matrix: np.ndarray, target_home: float,
                              target_draw: float, target_away: float) -> np.ndarray:
    """
    Herschaal de scorematrix zó dat de drie 1X2-gebieden (thuiswinst onder de
    diagonaal, gelijk op de diagonaal, uitwinst erboven) optellen tot de
    doelkansen — terwijl de VORM binnen elk gebied (welke exacte scores
    waarschijnlijk zijn) uit het model behouden blijft.

    Zo gieten we de markt (die alleen 1X2 kent) in een volledige matrix: de
    bookmaker bepaalt 'hoe waarschijnlijk wint wie', het Poisson-model bepaalt
    'en met welke score dan'.
    """
    m = matrix.astype(float).copy()
    cur_home = np.tril(m, -1).sum()
    cur_draw = np.trace(m)
    cur_away = np.triu(m, 1).sum()

    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            if i > j and cur_home > 0:        # thuiswinst-cellen
                m[i, j] *= target_home / cur_home
            elif i == j and cur_draw > 0:     # gelijkspel-cellen
                m[i, j] *= target_draw / cur_draw
            elif i < j and cur_away > 0:      # uitwinst-cellen
                m[i, j] *= target_away / cur_away
    total = m.sum()
    return m / total if total > 0 else m


def predict_match(ratings: dict, home: str, away: str, cal: Calibration,
                  neutral: bool = True, city: str | None = None,
                  odds_db: dict | None = None, weight_odds: float = 0.8) -> dict:
    """Volledige voorspelling voor een wedstrijd.

    Optionele uitbreidingen:
      - city: stadnaam -> past hoogte-correctie toe (alleen relevant in Mexico)
      - odds_db: dict met bookmaker-odds -> blendt model met markt
      - weight_odds: gewicht voor de odds in de blend (0..1)
    """
    rh = ratings.get(home, 1500.0)
    ra = ratings.get(away, 1500.0)

    # hoogte-correctie: pas Elo tijdelijk aan voor dit specifieke venue
    if city is not None:
        from altitude import altitude_adjustment
        rh = rh + altitude_adjustment(home, city)
        ra = ra + altitude_adjustment(away, city)

    lam_h, lam_a = expected_goals(rh, ra, cal, neutral)
    m = score_matrix(lam_h, lam_a, cal)
    model_probs = outcome_probs(m)

    # eventueel blenden met de markt
    final_probs = model_probs
    odds_used = None
    if odds_db is not None:
        from odds_fetcher import get_odds_probs, blend
        odds_used = get_odds_probs(odds_db, home, away)
        if odds_used is not None:
            final_probs = blend(model_probs, odds_used, weight_odds)
            # herschaal de VOLLEDIGE matrix naar de geblende 1X2, zodat de
            # heatmaps en het poule-advies ook de odds weerspiegelen
            m = rescale_matrix_to_outcome(m, *final_probs)

    sh, sa, p_score = most_likely_score(m)
    return {
        "home": home, "away": away,
        "lambda_home": round(lam_h, 2), "lambda_away": round(lam_a, 2),
        "p_home": round(final_probs[0], 3),
        "p_draw": round(final_probs[1], 3),
        "p_away": round(final_probs[2], 3),
        "p_home_model": round(model_probs[0], 3),
        "p_draw_model": round(model_probs[1], 3),
        "p_away_model": round(model_probs[2], 3),
        "has_odds": odds_used is not None,
        "score": f"{sh}-{sa}", "p_score": round(p_score, 3),
        "matrix": m,
    }
