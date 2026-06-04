"""
refresh_odds.py
---------------
Ververst de odds-cache met de meest recente bookmaker-odds.
Gebruik dit script vlak voor een speeldag — closing odds zijn het scherpst.

Stap 1: vraag een gratis API-key aan op https://the-odds-api.com/
        (500 requests/maand gratis, ruim genoeg voor een poule)

Stap 2: zet je key op ÉÉN van deze manieren:
        - omgevingsvariabele:           setx ODDS_API_KEY "jouw_sleutel"  (Windows, eenmalig)
        - of een tekstbestandje:        maak 'odds_api_key.txt' aan met alleen je sleutel erin
                                        (handig op een werk-pc waar je geen env vars wilt zetten)

Stap 3: draai dit script:
        python refresh_odds.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from odds_fetcher import fetch_odds, save_cache, cache_info


def get_api_key() -> str | None:
    # 1. omgevingsvariabele
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    # 2. lokaal bestandje
    p = Path("odds_api_key.txt")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    print("Huidige cache:", cache_info())
    print()

    key = get_api_key()
    if not key:
        print("Geen API-key gevonden.")
        print()
        print("Snelle route: maak een tekstbestand 'odds_api_key.txt' aan in deze map,")
        print("zet daar alleen je sleutel in (geen aanhalingstekens), en draai dit script opnieuw.")
        print("Sleutel aanvragen: https://the-odds-api.com/ (gratis, 500 calls/maand)")
        return 1

    print("Odds ophalen ...")
    try:
        odds_db = fetch_odds(key)
    except Exception as e:
        print(f"FOUT bij ophalen: {e}")
        return 2

    if not odds_db:
        print("Geen wedstrijden teruggekregen. Mogelijk staat het WK nog niet open in")
        print("The Odds API, of zijn er geen odds beschikbaar voor de huidige periode.")
        return 3

    save_cache(odds_db)
    print(f"OK: {len(odds_db)} wedstrijden opgeslagen in odds_cache.json")
    print()
    # toon een paar voorbeelden
    for i, (key, e) in enumerate(list(odds_db.items())[:3]):
        print(f"  {e['home']:25s} - {e['away']:25s}  "
              f"1={e['p_home']:.2f}  X={e['p_draw']:.2f}  2={e['p_away']:.2f}  ({e['bookmaker']})")
    if len(odds_db) > 3:
        print(f"  ... en nog {len(odds_db) - 3} wedstrijden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
