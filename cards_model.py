"""
cards_model.py
--------------
Schat de kaart-rubrieken voor de groepsfase op basis van historische
kaart-tarieven per land (uit cards_fetcher -> cards_cache.json).

Aanpak: per team trekken we voor de 3 groepswedstrijden het aantal gele en
rode kaarten uit een Poisson-verdeling met het team-tarief × 3. Daaruit:
  - 'meeste kaarten' : 1 pt per geel + 2 pt per rood; wie leidt het vaakst?
  - 'minste kaarten' : 5 pt bij 0 kaarten, 3 pt bij 1 geel, 1 pt bij 2 geel/1 rood.

LET OP: kaarten zijn ruisig en scheidsrechter-afhankelijk. Behandel dit als een
ruwe indicatie, niet als een sterke voorspelling.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_card_rates(path: str = "cards_cache.json") -> dict:
    """Laad kaart-tarieven. Lege dict als de cache er niet is."""
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload.get("rates", {})


def _fewest_points(yellow: int, red: int) -> int:
    """Scoreregel 'minste kaarten': 0 kaarten=5, 1 geel=3, 2 geel of 1 rood=1, anders 0."""
    if yellow == 0 and red == 0:
        return 5
    if yellow == 1 and red == 0:
        return 3
    if (yellow == 2 and red == 0) or (red == 1 and yellow == 0):
        return 1
    return 0


def simulate_cards(card_rates: dict, teams: list[str], n_sims: int = 20000,
                   matches_per_team: int = 3, seed: int = 7) -> dict:
    """
    Simuleer groepsfase-kaarten per team.
    Returns per team: verwachte gele/rode kaarten, verwachte 'kaartpunten'
    (1*geel+2*rood), kans om meeste-kaarten-koploper te zijn, en verwachte
    'minste kaarten'-punten.
    """
    rng = np.random.default_rng(seed)
    teams = [t for t in teams if t in card_rates]
    n = len(teams)

    y_rate = np.array([card_rates[t]["yellow_per_match"] * matches_per_team for t in teams])
    r_rate = np.array([card_rates[t]["red_per_match"] * matches_per_team for t in teams])

    sum_y = np.zeros(n); sum_r = np.zeros(n)
    sum_points = np.zeros(n)            # 1*geel + 2*rood
    most_leader = np.zeros(n)           # hoe vaak meeste kaartpunten
    fewest_points_sum = np.zeros(n)     # verwachte 'minste kaarten'-punten

    for _ in range(n_sims):
        y = rng.poisson(y_rate)
        r = rng.poisson(r_rate)
        pts = y + 2 * r
        sum_y += y; sum_r += r; sum_points += pts
        # meeste kaarten: koploper(s) op kaartpunten
        mx = pts.max()
        leaders = np.where(pts == mx)[0]
        most_leader[leaders] += 1 / len(leaders)
        # minste kaarten: punten per team deze sim
        fp = np.array([_fewest_points(int(y[i]), int(r[i])) for i in range(n)])
        fewest_points_sum += fp

    out = {}
    for i, t in enumerate(teams):
        out[t] = {
            "exp_yellow": sum_y[i] / n_sims,
            "exp_red": sum_r[i] / n_sims,
            "exp_card_points": sum_points[i] / n_sims,
            "p_most": most_leader[i] / n_sims,
            "exp_fewest_points": fewest_points_sum[i] / n_sims,
        }
    return out


def recommend_most_cards(card_sim: dict, top: int = 8):
    """Teams gerangschikt op kans om de meeste kaarten te krijgen."""
    return sorted(card_sim.items(), key=lambda kv: -kv[1]["p_most"])[:top]


def recommend_fewest_cards(card_sim: dict, top: int = 8):
    """Teams gerangschikt op verwachte 'minste kaarten'-punten (hoogste = beste gok)."""
    return sorted(card_sim.items(), key=lambda kv: -kv[1]["exp_fewest_points"])[:top]


if __name__ == "__main__":
    # test op synthetische tarieven (geen API nodig)
    demo_rates = {
        "Argentina": {"yellow_per_match": 2.6, "red_per_match": 0.15, "matches": 7},
        "Netherlands": {"yellow_per_match": 1.6, "red_per_match": 0.05, "matches": 7},
        "Japan": {"yellow_per_match": 1.1, "red_per_match": 0.02, "matches": 6},
        "Uruguay": {"yellow_per_match": 3.0, "red_per_match": 0.20, "matches": 7},
        "Norway": {"yellow_per_match": 1.3, "red_per_match": 0.04, "matches": 6},
    }
    sim = simulate_cards(demo_rates, list(demo_rates), n_sims=20000)
    print("Meeste kaarten (kans koploper):")
    for t, d in recommend_most_cards(sim):
        print(f"  {t:14s} {d['p_most']*100:5.1f}%  (verw. {d['exp_card_points']:.1f} kaartpunten)")
    print("\nMinste kaarten (verwachte punten):")
    for t, d in recommend_fewest_cards(sim):
        print(f"  {t:14s} {d['exp_fewest_points']:.2f} pt  "
              f"(verw. {d['exp_yellow']:.1f} geel, {d['exp_red']:.2f} rood)")
