"""
simulate.py
-----------
Monte-Carlo-simulatie van het volledige WK 2026 (48 teams, 12 groepen van 4).

Aanpak:
  - Groepen worden automatisch afgeleid uit de echte fixtures (teams die elkaar
    in de groepsfase treffen, zitten in dezelfde groep).
  - Per simulatie: speel elke groepswedstrijd uit door een score te trekken,
    bepaal de standen, kwalificeer top-2 per groep + de 8 beste nummers 3.
  - Knock-out: een geseede bracket (sterkste vs zwakste) — een nette benadering
    van het echte schema. Vervang `build_bracket` door het officiele schema
    zodra dat bekend is, als je het exact wilt.
  - Herhaal N keer en tel hoe vaak elk team elke fase haalt.

LET OP: dit schat TOERNOOI-kansen (wie wint, wie haalt de finale). Voor losse
groepswedstrijd-voorspellingen gebruik je model.predict_match() rechtstreeks.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np

from model import expected_goals, Calibration


def derive_groups(wc_fixtures) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str, str]]]]:
    """
    Leid de 12 groepen + bijbehorende fixtures af uit de wedstrijdlijst.

    Returns:
      groups   : dict label -> lijst van 4 teamnamen
      matches  : dict label -> lijst van (home, away, city) tuples voor die groep
    """
    adj = defaultdict(set)
    for r in wc_fixtures.itertuples():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)

    seen, groups, gid = set(), {}, 0
    for team in adj:
        if team in seen:
            continue
        members = sorted({team} | adj[team])
        if len(members) == 4:
            label = chr(ord("A") + gid)
            groups[label] = members
            seen.update(members)
            gid += 1

    # voeg per groep de werkelijke wedstrijden (met city) toe
    team_to_group = {t: g for g, members in groups.items() for t in members}
    matches: dict[str, list[tuple[str, str, str]]] = {g: [] for g in groups}
    for r in wc_fixtures.itertuples():
        g = team_to_group.get(r.home_team)
        if g is not None and team_to_group.get(r.away_team) == g:
            matches[g].append((r.home_team, r.away_team, r.city))
    return groups, matches


def _sample_score(rng, lam_h, lam_a):
    return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))


def _adjusted_rating(team: str, base_rating: float, city: str | None) -> float:
    """Pas hoogte-correctie toe als de wedstrijd in een Mexicaans venue is."""
    if city is None:
        return base_rating
    from altitude import altitude_adjustment
    return base_rating + altitude_adjustment(team, city)


def _simulate_match(rng, ratings, cal, home, away, knockout=False, city: str | None = None):
    """Simuleer 1 wedstrijd. Bij knock-out: geen gelijkspel (verlenging/penalty's)."""
    rh = _adjusted_rating(home, ratings.get(home, 1500), city)
    ra = _adjusted_rating(away, ratings.get(away, 1500), city)
    lam_h, lam_a = expected_goals(rh, ra, cal, neutral=True)
    gh, ga = _sample_score(rng, lam_h, lam_a)
    if knockout and gh == ga:
        # verlenging + penalty's: muntworp gewogen naar relatieve sterkte
        ph = lam_h / (lam_h + lam_a)
        return (home, gh, ga) if rng.random() < ph else (away, gh, ga)
    winner = home if gh > ga else (away if ga > gh else None)
    return (winner, gh, ga)


def _group_table(rng, ratings, cal, teams, fixtures):
    """Speel een hele groep met de ECHTE fixtures (incl. city voor hoogte)."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for h, a, city in fixtures:
        _, gh, ag = _simulate_match(rng, ratings, cal, h, a, city=city)
        gf[h] += gh; ga[h] += ag; gf[a] += ag; ga[a] += gh
        if gh > ag: pts[h] += 3
        elif ag > gh: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t], ratings.get(t, 1500)), reverse=True)
    return [{"team": t, "pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in ranked]


def _seed_positions(n: int) -> list[int]:
    """
    Standaard bracket-volgorde (1-indexed seeds) voor een veld van grootte n (macht van 2).
    Zorgt dat seed 1 en 2 elkaar pas in de finale kunnen treffen, maar legt het pad VAST.
    Bijv. n=4 -> [1, 4, 3, 2]: posities 1-4 en 3-2 spelen, winnaars in de finale.
    """
    order = [1, 2]
    while len(order) < n:
        size = len(order) * 2
        new = []
        for s in order:
            new.append(s)
            new.append(size + 1 - s)
        order = new
    return order


def build_bracket(rng, ratings, cal, qualifiers: list[str]) -> str:
    """
    Vaste geseede single-elimination knock-out (geen herseeding per ronde).
    Teams worden op Elo geseed en in een standaard bracket gezet; het pad ligt
    daarna vast, zodat sterke ploegen elkaar ook vroeg KUNNEN treffen.
    Geeft de uiteindelijke winnaar terug.
    """
    seeded = sorted(qualifiers, key=lambda t: ratings.get(t, 1500), reverse=True)
    n = 1
    while n * 2 <= len(seeded):
        n *= 2
    seeded = seeded[:n]
    # plaats teams in vaste bracketposities
    field = [seeded[s - 1] for s in _seed_positions(n)]
    while len(field) > 1:
        winners = []
        for i in range(0, len(field), 2):
            w, _, _ = _simulate_match(rng, ratings, cal, field[i], field[i + 1], knockout=True)
            winners.append(w)
        field = winners  # pad ligt vast, geen herseeding
    return field[0]


def simulate_tournament(ratings, cal, groups, matches, n_sims=20000, seed=42):
    """Draai het hele toernooi n_sims keer en tel fase-bereik per team."""
    rng = np.random.default_rng(seed)
    reach = defaultdict(lambda: {"last32": 0, "winner": 0})

    for _ in range(n_sims):
        thirds = []
        qualifiers = []
        for label, teams in groups.items():
            table = _group_table(rng, ratings, cal, teams, matches[label])
            qualifiers += [table[0]["team"], table[1]["team"]]   # top 2 plaatsen zich
            thirds.append((table[2]["pts"], table[2]["gd"], table[2]["gf"], table[2]["team"]))
        # 8 beste nummers 3
        thirds.sort(reverse=True)
        qualifiers += [t[3] for t in thirds[:8]]
        for t in qualifiers:
            reach[t]["last32"] += 1
        champion = build_bracket(rng, ratings, cal, qualifiers)
        reach[champion]["winner"] += 1

    out = []
    for team, d in reach.items():
        out.append({
            "team": team,
            "P_last32": round(d["last32"] / n_sims, 3),
            "P_winner": round(d["winner"] / n_sims, 3),
        })
    return sorted(out, key=lambda x: -x["P_winner"])
