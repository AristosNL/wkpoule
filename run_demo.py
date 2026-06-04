"""
run_demo.py
-----------
Eind-tot-eind demo met alle verfijningen:
  1. data inladen (echte internationale resultaten + WK 2026-fixtures)
  2. Elo-ratings berekenen
  3. scoremodel kalibreren
  4. odds-cache laden (indien aanwezig, na 'python refresh_odds.py')
  5. per groepswedstrijd voorspellingen printen, met:
       - hoogte-correctie voor Mexicaanse venues
       - blend met bookmaker-odds als die er zijn (anders alleen model)
  6. het hele toernooi simuleren -> kampioenskansen

Draai: python run_demo.py
Ververs odds: python refresh_odds.py
"""

from data_loader import load_international_results, split_played_and_fixtures, load_wc2026_fixtures
from elo import compute_elo
from model import calibrate, predict_match
from simulate import derive_groups, simulate_tournament
from odds_fetcher import load_cache, cache_info


def main():
    print("1/6  Data inladen ...")
    df = load_international_results()
    played, _ = split_played_and_fixtures(df)
    wc = load_wc2026_fixtures(df)

    print("2/6  Elo-ratings berekenen ...")
    ratings, history = compute_elo(played)

    print("3/6  Scoremodel kalibreren ...")
    cal = calibrate(history)
    print(f"     goals per Elo-punt = {cal.goals_per_elo:.5f} | basistotaal = {cal.base_total:.2f}")

    print(f"4/6  Odds-cache: {cache_info()}")
    odds_db = load_cache()
    if odds_db:
        print(f"     -> {len(odds_db)} wedstrijden met odds, model wordt geblend (80% odds / 20% model)")
    else:
        print("     -> geen odds gevonden, alleen model wordt gebruikt")
        print("        (tip: 'python refresh_odds.py' draaien voor live bookmaker-kansen)")

    print("\n5/6  Voorspellingen per groepswedstrijd:\n")
    print(f"  {'wedstrijd':45s} {'venue':18s} {'uitslag':>7s} {'1':>5s} {'X':>5s} {'2':>5s} {'odds?':>5s}")
    print("  " + "-" * 95)
    for r in wc.itertuples():
        p = predict_match(ratings, r.home_team, r.away_team, cal,
                          neutral=True, city=r.city, odds_db=odds_db if odds_db else None)
        match = f"{p['home']} - {p['away']}"
        mark = "+" if p["has_odds"] else " "
        print(f"  {match:45s} {r.city:18s} {p['score']:>7s} "
              f"{p['p_home']:>5.2f} {p['p_draw']:>5.2f} {p['p_away']:>5.2f} {mark:>5s}")

    print("\n6/6  Toernooi simuleren (Monte-Carlo) ...")
    groups, matches = derive_groups(wc)
    print(f"     {len(groups)} groepen gevonden")
    results = simulate_tournament(ratings, cal, groups, matches, n_sims=20000)
    print("\n  Kampioenskansen (top 15, alleen model):\n")
    print(f"  {'team':25s} {'P(winst)':>9s} {'P(laatste 32)':>14s}")
    print("  " + "-" * 50)
    for row in results[:15]:
        print(f"  {row['team']:25s} {row['P_winner']*100:>8.1f}% {row['P_last32']*100:>13.1f}%")


if __name__ == "__main__":
    main()
