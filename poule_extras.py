"""
poule_extras.py
---------------
Verrijkte simulatie die — naast de fasekansen — ook bijhoudt:
  - hoe vaak elk team op plek 1/2/3/4 van zijn groep eindigt  (voor groepstanden)
  - hoeveel goals een team gemiddeld voor/tegen krijgt in de groepsfase
  - hoe vaak een team de meeste goals voor / tegen van het toernooi heeft

Daarmee genereren we het optimale poule-advies voor de extra puntenrubrieken:
  1. groepstanden    : 2 pt exacte plek, 1 pt bij 1 plek afwijking
  2. finalisten      : 1/2/5/10/20/40 pt voor R32/R16/kwart/halve/finale/winnaar
  3a. meeste goals voor   : 1 pt per goal
  3b. meeste goals tegen  : 1 pt per goal

KAARTEN (3c/3d) ontbreken bewust: daar is geen data voor (zie uitleg in de chat).
"""

from __future__ import annotations
from collections import defaultdict
from itertools import permutations

import numpy as np

from simulate import _group_table, _simulate_match
from knockout import resolve_and_play


STAGE_POINTS_DEFAULT = {"last32": 1, "last16": 2, "quarter": 5,
                        "semi": 10, "final": 20, "winner": 40}
STAGE_COUNTS = {"last32": 32, "last16": 16, "quarter": 8,
                "semi": 4, "final": 2, "winner": 1}


def simulate_full(ratings, cal, groups, matches, n_sims=20000, seed=42):
    """Eén simulatieloop die alle benodigde statistieken in één keer verzamelt."""
    rng = np.random.default_rng(seed)
    stages = ["last32", "last16", "quarter", "semi", "final", "winner"]

    reach = defaultdict(lambda: {s: 0 for s in stages})
    pos_count = defaultdict(lambda: [0, 0, 0, 0])     # team -> tellingen plek 1..4
    team_group = {}                                   # team -> groepslabel (vast)
    gf_sum = defaultdict(float)                       # som goals voor (groepsfase)
    ga_sum = defaultdict(float)                       # som goals tegen
    most_gf = defaultdict(float)                      # hoe vaak meeste goals voor
    most_ga = defaultdict(float)                      # hoe vaak meeste goals tegen

    def sim_match(home, away, knockout=False, city=None):
        return _simulate_match(rng, ratings, cal, home, away, knockout=knockout, city=city)

    for _ in range(n_sims):
        winners, runners, thirds_by_group = {}, {}, {}
        third_rows = []
        sim_gf, sim_ga = {}, {}

        for label, teams in groups.items():
            table = _group_table(rng, ratings, cal, teams, matches[label])
            for pos, rowd in enumerate(table):
                t = rowd["team"]
                pos_count[t][pos] += 1
                team_group[t] = label
                gf = rowd["gf"]
                ga = rowd["gf"] - rowd["gd"]
                gf_sum[t] += gf
                ga_sum[t] += ga
                sim_gf[t] = gf
                sim_ga[t] = ga
            winners[label] = table[0]["team"]
            runners[label] = table[1]["team"]
            thirds_by_group[label] = table[2]["team"]
            third_rows.append((table[2]["pts"], table[2]["gd"], table[2]["gf"], label))

        # koploper goals voor / tegen deze simulatie (gelijke standen: eerlijk delen)
        max_gf = max(sim_gf.values())
        leaders_gf = [t for t, v in sim_gf.items() if v == max_gf]
        for t in leaders_gf:
            most_gf[t] += 1 / len(leaders_gf)
        max_ga = max(sim_ga.values())
        leaders_ga = [t for t, v in sim_ga.items() if v == max_ga]
        for t in leaders_ga:
            most_ga[t] += 1 / len(leaders_ga)

        # knock-out volgens officieel schema
        third_rows.sort(reverse=True)
        qualifying_third_groups = [r[3] for r in third_rows[:8]]
        reached = resolve_and_play(winners, runners, thirds_by_group,
                                   qualifying_third_groups, sim_match)
        for s in ["last32", "last16", "quarter", "semi", "final"]:
            for t in reached[s]:
                reach[t][s] += 1
        if reached["winner"] is not None:
            reach[reached["winner"]]["winner"] += 1

    # aggregeren
    stage_probs = []
    for team, d in reach.items():
        stage_probs.append({"team": team, **{f"P_{s}": d[s] / n_sims for s in stages}})
    stage_probs.sort(key=lambda x: -x["P_winner"])

    position_probs = defaultdict(dict)   # group -> {team: [p1,p2,p3,p4]}
    for team, counts in pos_count.items():
        g = team_group[team]
        position_probs[g][team] = [c / n_sims for c in counts]

    goals = {team: {"gf": gf_sum[team] / n_sims, "ga": ga_sum[team] / n_sims,
                    "p_most_gf": most_gf.get(team, 0) / n_sims,
                    "p_most_ga": most_ga.get(team, 0) / n_sims}
             for team in team_group}

    return {"stage_probs": stage_probs, "position_probs": dict(position_probs),
            "goals": goals}


# ----------------------------------------------------------------------------
# Advies-functies
# ----------------------------------------------------------------------------
def recommend_group_standing(team_pos_probs):
    """
    Kies de volgorde van 4 teams die de verwachte standen-punten maximaliseert.
    Scoreregel: 2 pt exacte plek, 1 pt bij precies 1 plek afwijking.
    Returns (geordende_teams, verwachte_punten).
    """
    teams = list(team_pos_probs)
    best, best_ev = None, -1.0
    for perm in permutations(teams):
        ev = 0.0
        for pos_idx, team in enumerate(perm):       # pos_idx 0 = plek 1
            p = team_pos_probs[team]
            ev += 2 * p[pos_idx]                     # exacte plek
            if pos_idx - 1 >= 0:
                ev += 1 * p[pos_idx - 1]             # 1 plek te hoog ingeschat
            if pos_idx + 1 < 4:
                ev += 1 * p[pos_idx + 1]             # 1 plek te laag ingeschat
        if ev > best_ev:
            best_ev, best = ev, perm
    return list(best), best_ev


def recommend_qualifiers(stage_probs, points=None):
    """Kies per fase de top-N teams op kans; geeft de lijsten + verwachte punten."""
    points = points or STAGE_POINTS_DEFAULT
    out = {}
    total_ev = 0.0
    for stage, count in STAGE_COUNTS.items():
        ranked = sorted(stage_probs, key=lambda x: -x[f"P_{stage}"])[:count]
        picks = [r["team"] for r in ranked]
        ev = points[stage] * sum(r[f"P_{stage}"] for r in ranked)
        total_ev += ev
        out[stage] = {"picks": picks, "ev": ev}
    return out, total_ev


def recommend_goal_leaders(goals):
    """Meest waarschijnlijke koploper goals voor en goals tegen (groepsfase)."""
    most_gf = max(goals.items(), key=lambda kv: kv[1]["p_most_gf"])
    most_ga = max(goals.items(), key=lambda kv: kv[1]["p_most_ga"])
    return {
        "meeste_goals_voor": {"team": most_gf[0], "kans": most_gf[1]["p_most_gf"],
                              "verwacht_aantal": most_gf[1]["gf"]},
        "meeste_goals_tegen": {"team": most_ga[0], "kans": most_ga[1]["p_most_ga"],
                               "verwacht_aantal": most_ga[1]["ga"]},
    }
