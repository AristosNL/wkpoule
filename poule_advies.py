"""
poule_advies.py
---------------
Print het volledige poule-advies: groepstanden, finalisten per fase en de
goaltotalen-rubrieken. Draai: python poule_advies.py
"""

from data_loader import load_international_results, split_played_and_fixtures, load_wc2026_fixtures
from elo import compute_elo
from model import calibrate
from simulate import derive_groups
from poule_extras import (simulate_full, recommend_group_standing,
                          recommend_qualifiers, recommend_goal_leaders,
                          STAGE_POINTS_DEFAULT, STAGE_COUNTS)


def main():
    print("Data laden + Elo + kalibratie ...")
    df = load_international_results()
    played, _ = split_played_and_fixtures(df)
    wc = load_wc2026_fixtures(df)
    ratings, history = compute_elo(played)
    cal = calibrate(history)
    groups, matches = derive_groups(wc)

    print("Verrijkte simulatie draaien (20.000 keer) ...\n")
    sim = simulate_full(ratings, cal, groups, matches, n_sims=20000)

    # ---- 1. Groepstanden ----
    print("=" * 60)
    print("1. GROEPSTANDEN-ADVIES  (2 pt exacte plek, 1 pt bij 1 plek afwijking)")
    print("=" * 60)
    totaal_standen = 0.0
    for label in sorted(sim["position_probs"]):
        order, ev = recommend_group_standing(sim["position_probs"][label])
        totaal_standen += ev
        print(f"\nGroep {label}  (verwacht {ev:.2f} pt):")
        for i, team in enumerate(order, 1):
            p = sim["position_probs"][label][team]
            print(f"   {i}. {team:22s}  (P plek {i} = {p[i-1]*100:4.1f}%)")
    print(f"\n  >> Verwacht totaal groepstanden: {totaal_standen:.1f} pt")

    # ---- 2. Finalisten per fase ----
    print("\n" + "=" * 60)
    print("2. FINALISTEN-ADVIES")
    print("=" * 60)
    quals, totaal_quals = recommend_qualifiers(sim["stage_probs"])
    labels = {"last32": "Laatste 32", "last16": "Achtste finale", "quarter": "Kwartfinale",
              "semi": "Halve finale", "final": "Finale", "winner": "Winnaar"}
    for stage in ["last32", "last16", "quarter", "semi", "final", "winner"]:
        info = quals[stage]
        pts = STAGE_POINTS_DEFAULT[stage]
        print(f"\n{labels[stage]} — kies {STAGE_COUNTS[stage]} ({pts} pt per correcte, "
              f"verwacht {info['ev']:.1f} pt):")
        # toon in regels van 4
        picks = info["picks"]
        for i in range(0, len(picks), 4):
            print("   " + ", ".join(picks[i:i+4]))
    print(f"\n  >> Verwacht totaal finalisten: {totaal_quals:.1f} pt")

    # ---- 3. Goaltotalen ----
    print("\n" + "=" * 60)
    print("3. GOALTOTALEN GROEPSFASE")
    print("=" * 60)
    gl = recommend_goal_leaders(sim["goals"])
    g = gl["meeste_goals_voor"]
    print(f"\nMeeste goals VOOR : {g['team']}  "
          f"(kans koploper {g['kans']*100:.0f}%, verwacht {g['verwacht_aantal']:.1f} goals)")
    g = gl["meeste_goals_tegen"]
    print(f"Meeste goals TEGEN: {g['team']}  "
          f"(kans koploper {g['kans']*100:.0f}%, verwacht {g['verwacht_aantal']:.1f} goals)")

    # ---- 4. Kaarten (alleen als cards_cache.json bestaat) ----
    from cards_model import (load_card_rates, simulate_cards,
                             recommend_most_cards, recommend_fewest_cards)
    print("\n" + "=" * 60)
    print("4. KAARTEN GROEPSFASE")
    print("=" * 60)
    card_rates = load_card_rates()
    if not card_rates:
        print("\nGeen kaartdata gevonden. Draai eerst 'python cards_fetcher.py'")
        print("(gratis API-Football-account vereist) om cards_cache.json te vullen.")
    else:
        teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
        csim = simulate_cards(card_rates, teams)
        print("\nMeeste kaarten (1 pt geel, 2 pt rood) — top 5:")
        for t, d in recommend_most_cards(csim, top=5):
            print(f"   {t:20s} kans koploper {d['p_most']*100:4.1f}%  "
                  f"(verw. {d['exp_card_points']:.1f} kaartpunten)")
        print("\nMinste kaarten (5/3/1 pt) — top 5 veiligste gokken:")
        for t, d in recommend_fewest_cards(csim, top=5):
            print(f"   {t:20s} verw. {d['exp_fewest_points']:.2f} pt  "
                  f"(verw. {d['exp_yellow']:.1f} geel, {d['exp_red']:.2f} rood)")


if __name__ == "__main__":
    main()
