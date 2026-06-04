"""
elo.py
------
Berekent Elo-ratings voor landenteams uit de volledige wedstrijdhistorie.

Variant: "World Football Elo" (zoals eloratings.net), met:
  - thuisvoordeel-bonus (niet bij neutraal terrein),
  - K-factor afhankelijk van toernooi-belang (WK telt zwaarder dan oefenduel),
  - goal-difference-multiplier (grote zeges schuiven de rating verder op).

De ratings zijn dynamisch: door over de tijd te lopen weegt recente vorm vanzelf
zwaarder. Dat vervangt de 'tijdsweging' uit het Dixon-Coles-model.
"""

from __future__ import annotations
from math import pow
import pandas as pd

START_RATING = 1500.0
K_BASE = 40.0
HOME_ADVANTAGE = 65.0  # Elo-punten; 0 op neutraal terrein

# Hoe zwaar telt een toernooi mee? (vermenigvuldigt de K-factor)
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 2.0,
    "FIFA World Cup qualification": 1.2,
    "UEFA Euro": 1.8,
    "UEFA Euro qualification": 1.1,
    "Copa América": 1.8,
    "African Cup of Nations": 1.5,
    "AFC Asian Cup": 1.5,
    "UEFA Nations League": 1.2,
    "Confederations Cup": 1.4,
    "Friendly": 1.0,
}
DEFAULT_WEIGHT = 1.1


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Grotere overwinningen wegen zwaarder, maar met afnemende meeropbrengst."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Verwachte 'score' (0..1) voor team A volgens de logistische Elo-formule."""
    return 1.0 / (1.0 + pow(10.0, -(rating_a - rating_b) / 400.0))


def compute_elo(played: pd.DataFrame) -> tuple[dict[str, float], list[dict]]:
    """
    Loop chronologisch door alle gespeelde wedstrijden en update ratings.

    Returns:
      ratings  : dict team -> huidige Elo
      history  : lijst met per wedstrijd het PRE-match ratingverschil en de uitslag
                 (gebruikt om het scoremodel te kalibreren, zie model.py)
    """
    ratings: dict[str, float] = {}
    history: list[dict] = []

    def get(team: str) -> float:
        return ratings.get(team, START_RATING)

    for row in played.itertuples():
        rh, ra = get(row.home_team), get(row.away_team)
        home_adv = 0.0 if row.neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra                     # pre-match ratingverschil
        we = expected_score(rh + home_adv, ra)        # verwachte score thuis

        if row.home_score > row.away_score:
            w = 1.0
        elif row.home_score == row.away_score:
            w = 0.5
        else:
            w = 0.0

        weight = TOURNAMENT_WEIGHT.get(row.tournament, DEFAULT_WEIGHT)
        k = K_BASE * weight * _goal_diff_multiplier(row.home_score - row.away_score)

        # bewaar pre-match info voor kalibratie van het scoremodel
        history.append({
            "date": row.date,
            "dr": dr,
            "supremacy": row.home_score - row.away_score,
            "total_goals": row.home_score + row.away_score,
        })

        ratings[row.home_team] = rh + k * (w - we)
        ratings[row.away_team] = ra + k * ((1 - w) - (1 - we))

    return ratings, history


if __name__ == "__main__":
    from data_loader import load_international_results, split_played_and_fixtures
    played, _ = split_played_and_fixtures(load_international_results())
    ratings, _ = compute_elo(played)
    top = sorted(ratings.items(), key=lambda x: -x[1])[:20]
    print("Top 20 Elo:")
    for team, r in top:
        print(f"  {team:25s} {r:6.0f}")
