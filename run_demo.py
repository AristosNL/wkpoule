"""
run_demo.py
-----------
Eind-tot-eind voorspeller, afgestemd op jouw poule-scoreregels (4/6/10 cumulatief).
"""

from data_loader import load_international_results, split_played_and_fixtures, load_wc2026_fixtures
from elo import compute_elo
from model import calibrate, predict_match
from simulate import derive_groups, simulate_tournament
from odds_fetcher import load_cache, cache_info
from poule_strategy import ScoringRules, optimal_prediction, expected_points
import numpy as np


# === Pas dit aan als jouw poule andere regels heeft ===
RULES = ScoringRules(exact=10, goal_diff=6, winner=4, cumulative=False)


def main():
    print("1/6  Data inladen ...")
    df = load_international_results()
    played, _ = split_played_and_fixtures(df)
    wc = load_wc2026_fixtures(df)

    print("2/6  Elo-ratings berekenen ...")
    ratings, history = compute_elo(played)

    print("3/6  Scoremodel kalibreren ...")
    cal = calibrate(history)
    print(f"     goals per Elo-punt = {cal.goals_per_elo:.5f} | "
          f"totaal = {cal.total_intercept:.2f} + {cal.total_slope:.5f}*|Elo-gap|")

    print(f"4/6  Odds-cache: {cache_info()}")
    odds_db = load_cache()
    if odds_db:
        print(f"     -> {len(odds_db)} wedstrijden met odds, model wordt geblend (80% odds / 20% model)")
    else:
        print("     -> geen odds gevonden, alleen model wordt gebruikt")
        print("        (tip: 'python refresh_odds.py' draaien voor live bookmaker-kansen)")

    print(f"\n5/6  Voorspellingen per groepswedstrijd")
    print(f"     scoreregels: winnaar={RULES.winner}, doelsaldo={RULES.goal_diff}, exact={RULES.exact}"
          f" ({'cumulatief' if RULES.cumulative else 'beste laag telt'})\n")
    print(f"  {'wedstrijd':45s} {'venue':16s} {'modus':>6s} {'POULE':>6s} {'EV':>5s} "
          f"{'1':>5s} {'X':>5s} {'2':>5s} {'odds':>5s}")
    print("  " + "-" * 110)

    totaal_ev = 0.0
    for r in wc.itertuples():
        p = predict_match(ratings, r.home_team, r.away_team, cal,
                          neutral=True, city=r.city, odds_db=odds_db if odds_db else None)
        m = p["matrix"]
        mi, mj = np.unravel_index(np.argmax(m), m.shape)
        oh, oa, ev = optimal_prediction(m, RULES)
        totaal_ev += ev
        match = f"{p['home']} - {p['away']}"
        odds_mark = "+" if p["has_odds"] else " "
        print(f"  {match:45s} {r.city:16s} {f'{mi}-{mj}':>6s} {f'{oh}-{oa}':>6s} {ev:>5.2f} "
              f"{p['p_home']:>5.2f} {p['p_draw']:>5.2f} {p['p_away']:>5.2f} {odds_mark:>5s}")

    print("  " + "-" * 110)
    print(f"  Verwachte TOTAAL-punten over de groepsfase met poule-optimale gokken: {totaal_ev:.1f}")
    print(f"  (vul de kolom 'POULE' in je formulier)\n")

    print("6/6  Toernooi simuleren (officieel R32-schema, Monte-Carlo) ...")
    groups, matches = derive_groups(wc)
    print(f"     {len(groups)} groepen gevonden")
    results = simulate_tournament(ratings, cal, groups, matches, n_sims=20000)
    print("\n  Kansen per fase (top 15):\n")
    print(f"  {'team':22s} {'titel':>6s} {'finale':>7s} {'halve':>6s} {'kwart':>6s} {'R16':>6s}")
    print("  " + "-" * 60)
    for row in results[:15]:
        print(f"  {row['team']:22s} {row['P_winner']*100:5.1f}% {row['P_final']*100:6.1f}% "
              f"{row['P_semi']*100:5.1f}% {row['P_quarter']*100:5.1f}% {row['P_last16']*100:5.1f}%")


if __name__ == "__main__":
    main()
