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


# Officiële WK 2026-groepsindeling (FIFA), niet de alfabetische volgorde uit de fixtures.
FIFA_GROUPS = {
    "A": frozenset({"Mexico", "South Africa", "South Korea", "Czech Republic"}),
    "B": frozenset({"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"}),
    "C": frozenset({"Brazil", "Morocco", "Haiti", "Scotland"}),
    "D": frozenset({"United States", "Paraguay", "Australia", "Turkey"}),
    "E": frozenset({"Germany", "Curaçao", "Ivory Coast", "Ecuador"}),
    "F": frozenset({"Netherlands", "Japan", "Sweden", "Tunisia"}),
    "G": frozenset({"Belgium", "Egypt", "Iran", "New Zealand"}),
    "H": frozenset({"Spain", "Cape Verde", "Saudi Arabia", "Uruguay"}),
    "I": frozenset({"France", "Senegal", "Iraq", "Norway"}),
    "J": frozenset({"Argentina", "Algeria", "Austria", "Jordan"}),
    "K": frozenset({"Portugal", "DR Congo", "Uzbekistan", "Colombia"}),
    "L": frozenset({"England", "Croatia", "Ghana", "Panama"}),
}


def derive_groups(wc_fixtures) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str, str]]]]:
    """
    Bepaal de 12 groepen + bijbehorende fixtures met de OFFICIËLE FIFA-labels.
    De fixture-volgorde bepaalt niet meer welke groep welke letter krijgt.
    """
    adj = defaultdict(set)
    for r in wc_fixtures.itertuples():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)

    # bepaal de feitelijke groepen (4 teams die elkaar treffen)
    seen, raw_groups = set(), []
    for team in adj:
        if team in seen:
            continue
        members = frozenset({team} | adj[team])
        if len(members) == 4:
            raw_groups.append(members)
            seen.update(members)

    # match elke gevonden groep tegen de officiële FIFA-indeling
    groups: dict[str, list[str]] = {}
    for label, official in FIFA_GROUPS.items():
        for rg in raw_groups:
            if rg == official:
                groups[label] = sorted(rg)
                break
        else:
            # zou niet mogen gebeuren als de fixtures kloppen
            print(f"  ! waarschuwing: officiële groep {label} niet gevonden in fixtures")

    # fixtures per groep met de officiële labels
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


def simulate_tournament(ratings, cal, groups, matches, n_sims=20000, seed=42):
    """
    Draai het hele toernooi n_sims keer met het OFFICIËLE WK 2026-schema en tel
    per team hoe vaak het elke fase haalt.

    De knock-out volgt het echte R32-schema (zie knockout.py): groepswinnaars
    tegen nummers 3, runners-up tegen elkaar, geen groepsgenoten in de R32.
    """
    from knockout import resolve_and_play

    rng = np.random.default_rng(seed)
    stages = ["last32", "last16", "quarter", "semi", "final", "winner"]
    reach = defaultdict(lambda: {s: 0 for s in stages})

    def sim_match(home, away, knockout=False, city=None):
        return _simulate_match(rng, ratings, cal, home, away, knockout=knockout, city=city)

    for _ in range(n_sims):
        winners, runners, thirds_by_group = {}, {}, {}
        third_rows = []
        for label, teams in groups.items():
            table = _group_table(rng, ratings, cal, teams, matches[label])
            winners[label] = table[0]["team"]
            runners[label] = table[1]["team"]
            thirds_by_group[label] = table[2]["team"]
            third_rows.append((table[2]["pts"], table[2]["gd"], table[2]["gf"], label))

        # 8 beste nummers 3 (op punten, doelsaldo, goals)
        third_rows.sort(reverse=True)
        qualifying_third_groups = [r[3] for r in third_rows[:8]]

        reached = resolve_and_play(winners, runners, thirds_by_group,
                                   qualifying_third_groups, sim_match)
        for s in ["last32", "last16", "quarter", "semi", "final"]:
            for t in reached[s]:
                reach[t][s] += 1
        if reached["winner"] is not None:
            reach[reached["winner"]]["winner"] += 1

    out = []
    for team, d in reach.items():
        out.append({
            "team": team,
            "P_last32": round(d["last32"] / n_sims, 3),
            "P_last16": round(d["last16"] / n_sims, 3),
            "P_quarter": round(d["quarter"] / n_sims, 3),
            "P_semi": round(d["semi"] / n_sims, 3),
            "P_final": round(d["final"] / n_sims, 3),
            "P_winner": round(d["winner"] / n_sims, 3),
        })
    return sorted(out, key=lambda x: -x["P_winner"])
