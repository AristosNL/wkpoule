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


def simulate_full(ratings, cal, groups, matches, n_sims=20000, seed=42,
                  odds_db=None, weight_odds=0.0, outright_probs=None,
                  card_rates=None):
    """Eén simulatieloop die alle benodigde statistieken in één keer verzamelt.

    Groepsfase: geblende Poisson-matrix (odds_db + weight_odds).
    Knock-out  : per-wedstrijd odds → Bradley-Terry (met Elo-scoreschatting)
                 → puur Elo. card_rates optioneel voor kaarten per ronde.
    """
    from simulate import precompute_fixture_distributions
    rng = np.random.default_rng(seed)
    stages = ["last32", "last16", "quarter", "semi", "final", "winner"]

    reach = defaultdict(lambda: {s: 0 for s in stages})
    pos_count = defaultdict(lambda: [0, 0, 0, 0])     # team -> tellingen plek 1..4
    team_group = {}                                   # team -> groepslabel (vast)
    gf_sum = defaultdict(float)                       # som goals voor (groepsfase)
    ga_sum = defaultdict(float)                       # som goals tegen
    most_gf = defaultdict(float)                      # hoe vaak meeste goals voor
    most_ga = defaultdict(float)                      # hoe vaak meeste goals tegen
    # bracket-tracking: per (ronde, match_idx, side) -> {team: count}
    bracket_pos = defaultdict(lambda: defaultdict(int))
    # per-ronde goal/kaart-tracking (onvoorwaardelijke kansen: delen door n_sims)
    rnd_gf    = defaultdict(float)   # (rn, team) -> sum goals voor
    rnd_ga    = defaultdict(float)   # (rn, team) -> sum goals tegen
    rnd_n     = defaultdict(int)     # (rn, team) -> aantaldeelnames
    rnd_most_gf  = defaultdict(float)
    rnd_most_ga  = defaultdict(float)
    rnd_cpts     = defaultdict(float) if card_rates else None
    rnd_fewest   = defaultdict(float) if card_rates else None
    rnd_most_c   = defaultdict(float) if card_rates else None
    rng_c = np.random.default_rng(seed + 31337) if card_rates else None

    # eenmalige precompute van de 72 groeps-distributies
    fixture_dists = precompute_fixture_distributions(ratings, cal, matches,
                                                     odds_db=odds_db,
                                                     weight_odds=weight_odds)

    def sim_match(home, away, knockout=False, city=None):
        """Prioriteitsketen voor knockout:
           1. Per-wedstrijd odds in odds_db (bv. Nederland-Marokko)
           2. Bradley-Terry op outright_probs (bookmaker titelkansen)
           3. Puur Elo
        """
        if knockout:
            # --- prioriteit 1: per-wedstrijd odds ---
            if odds_db:
                from odds_fetcher import get_odds_probs
                from model import (score_matrix, rescale_matrix_to_outcome,
                                   outcome_probs, expected_goals as _eg)
                from odds_fetcher import blend as odds_blend
                op = get_odds_probs(odds_db, home, away)
                if op is not None:
                    lam_h, lam_a = _eg(
                        ratings.get(home, 1500), ratings.get(away, 1500), cal)
                    m = score_matrix(lam_h, lam_a, cal)
                    if weight_odds < 1.0:
                        model_p = outcome_probs(m)
                        target = odds_blend(model_p, op, weight_odds)
                    else:
                        target = op
                    m = rescale_matrix_to_outcome(m, *target)
                    flat = m.flatten(); flat = flat / flat.sum()
                    idx = rng.choice(len(flat), p=flat)
                    gh, ga = divmod(int(idx), m.shape[1])
                    if gh == ga:
                        # verlenging (30 min): extra goals uit het Elo-model
                        from simulate import _play_extra_time
                        gh, ga = _play_extra_time(rng, gh, ga, lam_h, lam_a)
                        if gh == ga:
                            # penalty's: winnaar uit de odds, score blijft gelijk
                            p_hw = target[0] / (target[0] + target[2])
                            winner = home if rng.random() < p_hw else away
                        else:
                            winner = home if gh > ga else away
                    else:
                        winner = home if gh > ga else away
                    return (winner, gh, ga)
            # --- prioriteit 2: Bradley-Terry op outright-kansen ---
            if outright_probs:
                pa = outright_probs.get(home)
                pb = outright_probs.get(away)
                if pa is not None and pb is not None and (pa + pb) > 0:
                    from model import expected_goals as _eg
                    from simulate import _play_extra_time
                    lam_h, lam_a = _eg(ratings.get(home, 1500),
                                       ratings.get(away, 1500), cal)
                    gh, ga = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                    if gh == ga:                      # verlenging
                        gh, ga = _play_extra_time(rng, gh, ga, lam_h, lam_a)
                    bt_winner = home if rng.random() < pa / (pa + pb) else away
                    if gh == ga:
                        # gelijk na verlenging -> penalty's (BT bepaalt), score blijft gelijk
                        return (bt_winner, gh, ga)
                    # beslissende stand -> hogere score naar de BT-winnaar
                    hi, lo = max(gh, ga), min(gh, ga)
                    return (home, hi, lo) if bt_winner == home else (away, lo, hi)
        # --- prioriteit 3 / groepswedstrijden: puur Elo ---
        return _simulate_match(rng, ratings, cal, home, away,
                               knockout=knockout, city=city)

    for _ in range(n_sims):
        winners, runners, thirds_by_group = {}, {}, {}
        third_rows = []
        sim_gf, sim_ga = {}, {}

        for label, teams in groups.items():
            table = _group_table(rng, ratings, cal, teams, matches[label],
                                 fixture_dists=fixture_dists)
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

        # bracket per ronde aggregeren
        for round_data in reached.get("bracket", []):
            rn = round_data["round"]
            sim_gf_rn: dict = {}
            sim_ga_rn: dict = {}
            sim_c_rn:  dict = {} if card_rates else None

            for mi, entry in enumerate(round_data["matches"]):
                h, a, w, gh, ga = entry
                bracket_pos[(rn, mi, "home")][h] += 1
                bracket_pos[(rn, mi, "away")][a] += 1
                bracket_pos[(rn, mi, "winner")][w] += 1
                # goals
                for team, gf_v, ga_v in [(h, gh, ga), (a, ga, gh)]:
                    sim_gf_rn[team] = sim_gf_rn.get(team, 0) + gf_v
                    sim_ga_rn[team] = sim_ga_rn.get(team, 0) + ga_v
                    rnd_n[(rn, team)] += 1
                    rnd_gf[(rn, team)] += gf_v
                    rnd_ga[(rn, team)] += ga_v
                # kaarten
                if card_rates:
                    for team in (h, a):
                        cr = card_rates.get(team)
                        if cr:
                            yc = int(rng_c.poisson(cr["yellow_per_match"]))
                            rc = int(rng_c.poisson(cr["red_per_match"]))
                            pts = yc + 2 * rc
                            sim_c_rn[team] = sim_c_rn.get(team, 0) + pts
                            rnd_cpts[(rn, team)] += pts
                            # fewest scoring
                            fp = (5 if yc == 0 and rc == 0 else
                                  3 if yc == 1 and rc == 0 else
                                  1 if (yc == 2 and rc == 0) or (rc == 1 and yc == 0) else 0)
                            rnd_fewest[(rn, team)] += fp
            # koplopers goals
            for d, acc in [(sim_gf_rn, rnd_most_gf), (sim_ga_rn, rnd_most_ga)]:
                if d:
                    mx = max(d.values())
                    lds = [t for t, v in d.items() if v == mx]
                    for t in lds:
                        acc[(rn, t)] += 1 / len(lds)
            # koplopers kaarten
            if card_rates and sim_c_rn:
                mx = max(sim_c_rn.values())
                lds = [t for t, v in sim_c_rn.items() if v == mx]
                for t in lds:
                    rnd_most_c[(rn, t)] += 1 / len(lds)

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

    # modale bracket: per slot het meest voorkomende team + zijn waarschijnlijkheid
    modal_bracket = []
    round_order = ["R32", "R16", "QF", "SF", "F"]
    round_sizes = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}
    for rn in round_order:
        round_matches = []
        for mi in range(round_sizes[rn]):
            def pick(side):
                d = bracket_pos.get((rn, mi, side), {})
                if not d:
                    return ("?", 0.0)
                team, count = max(d.items(), key=lambda kv: kv[1])
                return (team, count / n_sims)
            home, p_h = pick("home")
            away, p_a = pick("away")
            winner, p_w = pick("winner")
            round_matches.append({"home": home, "away": away, "winner": winner,
                                  "p_home": p_h, "p_away": p_a, "p_winner": p_w})
        modal_bracket.append({"round": rn, "matches": round_matches})

    # ---- round_stats: per ronde top-teams per categorie ----
    round_stats: dict = {}
    for rn in ["R32", "R16", "QF", "SF", "F"]:
        teams_rn = {t for (r, t) in rnd_n if r == rn}
        if not teams_rn:
            continue
        rs: dict = {}
        for t in teams_rn:
            n = rnd_n.get((rn, t), 0)
            if not n:
                continue
            entry: dict = {
                "p_appears":  n / n_sims,
                "exp_gf":     rnd_gf.get((rn, t), 0) / n_sims,
                "exp_ga":     rnd_ga.get((rn, t), 0) / n_sims,
                "p_most_gf":  rnd_most_gf.get((rn, t), 0) / n_sims,
                "p_most_ga":  rnd_most_ga.get((rn, t), 0) / n_sims,
            }
            if card_rates:
                entry["exp_cpts"]       = rnd_cpts.get((rn, t), 0) / n_sims
                entry["p_most_cards"]   = rnd_most_c.get((rn, t), 0) / n_sims
                entry["exp_fewest_pts"] = rnd_fewest.get((rn, t), 0) / n_sims
            rs[t] = entry
        round_stats[rn] = rs

    return {"stage_probs": stage_probs, "position_probs": dict(position_probs),
            "goals": goals, "bracket": modal_bracket, "round_stats": round_stats}


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


def _knockout_after_et_matrix(home, away, odds_db, ratings, cal, weight_odds,
                              use_totals=True):
    """Bouw de score-kansmatrix NA verlenging voor één knock-outwedstrijd:
    90-min-matrix → (optioneel) totaal herschaald naar de over/under-markt →
    odds-blend op 1X2 → 90-min-gelijkspelen herverdeeld over de stand na verlenging."""
    from math import exp, factorial
    from model import (score_matrix, rescale_matrix_to_outcome, outcome_probs,
                       expected_goals as _eg)
    from odds_fetcher import get_odds_probs, blend as odds_blend, get_implied_total
    from simulate import ET_GOAL_FACTOR

    def _pois(k, lam):
        return exp(-lam) * lam ** k / factorial(k)

    lam_h, lam_a = _eg(ratings.get(home, 1500), ratings.get(away, 1500), cal)
    # herschaal het verwachte totaal naar de over/under-markt (verhouding behouden)
    if use_totals:
        T = get_implied_total(odds_db, home, away)
        if T is not None and (lam_h + lam_a) > 0:
            f = T / (lam_h + lam_a)
            lam_h, lam_a = lam_h * f, lam_a * f

    M = score_matrix(lam_h, lam_a, cal)
    op = get_odds_probs(odds_db, home, away)
    if op is not None:
        target = op if weight_odds >= 1.0 else odds_blend(outcome_probs(M), op, weight_odds)
        M = rescale_matrix_to_outcome(M, *target)

    n = M.shape[0]
    et_h = ET_GOAL_FACTOR * lam_h / 3.0
    et_a = ET_GOAL_FACTOR * lam_a / 3.0
    M_adj = M.copy()
    for k in range(n):
        pk = M[k, k]
        if pk <= 0:
            continue
        M_adj[k, k] = 0.0
        for di in range(0, n - k):
            ph = _pois(di, et_h)
            for dj in range(0, n - k):
                M_adj[k + di, k + dj] += pk * ph * _pois(dj, et_a)
    s = M_adj.sum()
    if s > 0:
        M_adj /= s
    return M_adj


def knockout_match_predictions(odds_db, ratings, cal, weight_odds, rules, use_totals=True):
    """
    Voor elke BEKENDE knock-outwedstrijd (round == 'knockout' in odds_db) de
    optimale score-voorspelling die de verwachte poulepunten maximaliseert.
    Houdt rekening met de odds-blend en de verlenging-correctie.
    """
    from model import outcome_probs
    from poule_strategy import optimal_prediction, top_n_predictions

    out = []
    for key, info in odds_db.items():
        if info.get("round") != "knockout":
            continue
        h, a = info["home"], info["away"]
        M_adj = _knockout_after_et_matrix(h, a, odds_db, ratings, cal, weight_odds, use_totals)
        gh, ga, ev = optimal_prediction(M_adj, rules)
        probs = outcome_probs(M_adj)
        alts = top_n_predictions(M_adj, rules, n=3)
        out.append({
            "home": h, "away": a,
            "pred_h": gh, "pred_a": ga, "ev": round(ev, 2),
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "alts": [(x, y, round(e, 2)) for x, y, e in alts],
            "bookmaker": info.get("bookmaker", ""),
            "commence_time": info.get("commence_time", ""),
        })
    out.sort(key=lambda r: r["commence_time"])
    return out


def knockout_round_stats(odds_db, ratings, cal, weight_odds, card_rates=None, use_totals=True,
                         n_sims=20000, seed=11):
    """
    Doelpunten (voor/tegen) en kaarten per team voor de BEKENDE knock-outronde,
    op basis van de werkelijke matchups uit odds_db. Monte-Carlo over de
    na-verlenging-scoreverdeling per wedstrijd.

    Returns dict met round-label, aantal wedstrijden en per team:
      exp_gf, exp_ga, p_most_gf, p_most_ga (+ kaartvelden als card_rates gegeven).
    """
    import numpy as np
    from collections import defaultdict

    matches = [(v["home"], v["away"]) for v in odds_db.values()
               if v.get("round") == "knockout"]
    if not matches:
        return None

    # precompute per-match cumulatieve verdeling om uit te samplen
    dists = []
    for h, a in matches:
        M = _knockout_after_et_matrix(h, a, odds_db, ratings, cal, weight_odds, use_totals)
        flat = M.flatten()
        flat = flat / flat.sum()
        dists.append((h, a, flat, M.shape[1]))

    rng = np.random.default_rng(seed)
    teams = sorted({t for m in matches for t in m})
    gf = defaultdict(float); ga = defaultdict(float)
    most_gf = defaultdict(float); most_ga = defaultdict(float)
    cpts = defaultdict(float); most_c = defaultdict(float); fewest = defaultdict(float)

    for _ in range(n_sims):
        sim_gf, sim_ga, sim_c = {}, {}, {}
        for h, a, flat, ncol in dists:
            idx = rng.choice(len(flat), p=flat)
            gh, ag = divmod(int(idx), ncol)
            sim_gf[h], sim_ga[h] = gh, ag
            sim_gf[a], sim_ga[a] = ag, gh
            gf[h] += gh; ga[h] += ag; gf[a] += ag; ga[a] += gh
            if card_rates:
                for team in (h, a):
                    cr = card_rates.get(team)
                    if cr:
                        yc = int(rng.poisson(cr["yellow_per_match"]))
                        rc = int(rng.poisson(cr["red_per_match"]))
                        sim_c[team] = yc + 2 * rc
                        cpts[team] += yc + 2 * rc
                        fewest[team] += (5 if yc == 0 and rc == 0 else
                                         3 if yc == 1 and rc == 0 else
                                         1 if (yc == 2 and rc == 0) or (rc == 1 and yc == 0) else 0)
        for d, acc in [(sim_gf, most_gf), (sim_ga, most_ga)]:
            mx = max(d.values())
            lds = [t for t, v in d.items() if v == mx]
            for t in lds:
                acc[t] += 1 / len(lds)
        if card_rates and sim_c:
            mx = max(sim_c.values())
            lds = [t for t, v in sim_c.items() if v == mx]
            for t in lds:
                most_c[t] += 1 / len(lds)

    res = {}
    for t in teams:
        res[t] = {"exp_gf": gf[t] / n_sims, "exp_ga": ga[t] / n_sims,
                  "p_most_gf": most_gf[t] / n_sims, "p_most_ga": most_ga[t] / n_sims}
        if card_rates:
            res[t].update({"exp_cpts": cpts[t] / n_sims,
                           "p_most_cards": most_c[t] / n_sims,
                           "exp_fewest_pts": fewest[t] / n_sims})
    label = {16: "Laatste 32", 8: "Achtste finale", 4: "Kwartfinale",
             2: "Halve finale", 1: "Finale"}.get(len(matches), f"{len(matches)} duels")
    return {"round": label, "n_matches": len(matches), "teams": res,
            "has_cards": bool(card_rates)}
