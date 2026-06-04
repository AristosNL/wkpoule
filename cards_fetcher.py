"""
cards_fetcher.py
----------------
Haalt historische kaart-tarieven per WK 2026-land op via API-Football (gratis).

Stappenplan:
  1. Gratis account op https://www.api-football.com/ (geen creditcard).
     Je vindt je sleutel in het dashboard.
  2. Zet je sleutel in een tekstbestand 'apifootball_key.txt' in deze map
     (alleen de sleutel), of in de omgevingsvariabele APIFOOTBALL_KEY.
  3. Draai:  python cards_fetcher.py
     -> resultaat komt in cards_cache.json (kaarten per wedstrijd per land).

Let op: de gratis tier is ~100 requests/dag. Het script cachet team-ID's en
tarieven, zodat een herhaalde run nauwelijks nieuwe requests kost.

Hoe het werkt: voor elke (competitie, seizoen) in COMPETITIONS halen we per team
de teamstatistieken op. Daarin staan gele/rode kaarten (per minuutblok) plus het
aantal gespeelde wedstrijden. Daaruit volgt: kaarten per wedstrijd.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

import requests

API_BASE = "https://v3.football.api-sports.io"

# (league_id, season) om kaart-data over te aggregeren. Standaard: WK 2022 (league 1).
# Meer toevoegen = betere tarieven, maar meer API-calls. League-ID's vind je via
# het /leagues endpoint of de API-Football docs (bv. Nations League = 5).
COMPETITIONS = [(1, 2022)]

# Onze datasetnaam -> zoekterm voor API-Football (alleen waar ze afwijken).
SEARCH_NAME = {
    "United States": "USA",
    "South Korea": "South Korea",
    "Czech Republic": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia",
    "Cape Verde": "Cape Verde Islands",
}


def _headers(api_key: str) -> dict:
    return {"x-apisports-key": api_key}


def _get(api_key: str, path: str, params: dict) -> dict:
    resp = requests.get(f"{API_BASE}{path}", headers=_headers(api_key), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_api_key() -> str | None:
    key = os.environ.get("APIFOOTBALL_KEY")
    if key:
        return key.strip()
    p = Path("apifootball_key.txt")
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def resolve_team_ids(api_key: str, team_names: list[str],
                     cache: str = "team_ids_cache.json") -> dict[str, int]:
    """Zoek de API-Football team-ID per land. Cachet zodat het maar één keer kost."""
    known = {}
    if Path(cache).exists():
        known = json.loads(Path(cache).read_text(encoding="utf-8"))

    for name in team_names:
        if name in known:
            continue
        query = SEARCH_NAME.get(name, name)
        try:
            data = _get(api_key, "/teams", {"search": query})
            results = data.get("response", [])
            # bij voorkeur een nationaal team
            nat = [r for r in results if r.get("team", {}).get("national")]
            pick = (nat or results)
            if pick:
                known[name] = pick[0]["team"]["id"]
            else:
                known[name] = None
        except Exception as e:
            print(f"  ! kon team-ID niet ophalen voor {name}: {e}")
            known[name] = None
        time.sleep(0.2)
    Path(cache).write_text(json.dumps(known, indent=2, ensure_ascii=False), encoding="utf-8")
    return known


def _sum_card_bucket(card_section: dict) -> int:
    """Tel de 'total'-waarden over alle minuutblokken op (geel of rood)."""
    total = 0
    for bucket in card_section.values():
        if isinstance(bucket, dict) and bucket.get("total") is not None:
            total += bucket["total"]
    return total


def fetch_card_rates(api_key: str, team_ids: dict[str, int],
                     competitions=COMPETITIONS, cache: str = "cards_cache.json") -> dict:
    """
    Haal per team de kaarten op en bereken geel/rood per wedstrijd, geaggregeerd
    over de opgegeven competities. Schrijft cards_cache.json.
    """
    rates = {}
    for name, tid in team_ids.items():
        if not tid:
            continue
        yellow_tot = red_tot = matches_tot = 0
        for (lid, season) in competitions:
            try:
                data = _get(api_key, "/teams/statistics",
                            {"league": lid, "season": season, "team": tid})
                r = data.get("response") or {}
                cards = r.get("cards", {}) or {}
                played = (r.get("fixtures", {}).get("played", {}) or {}).get("total") or 0
                if played:
                    yellow_tot += _sum_card_bucket(cards.get("yellow", {}) or {})
                    red_tot += _sum_card_bucket(cards.get("red", {}) or {})
                    matches_tot += played
            except Exception as e:
                print(f"  ! stats mislukt voor {name} (league {lid}, {season}): {e}")
            time.sleep(0.2)
        if matches_tot:
            rates[name] = {
                "yellow_per_match": round(yellow_tot / matches_tot, 3),
                "red_per_match": round(red_tot / matches_tot, 3),
                "matches": matches_tot,
            }

    # teams zonder data: vul aan met het gemiddelde tarief
    if rates:
        avg_y = sum(v["yellow_per_match"] for v in rates.values()) / len(rates)
        avg_r = sum(v["red_per_match"] for v in rates.values()) / len(rates)
    else:
        avg_y, avg_r = 2.0, 0.1   # redelijke defaults als er niets is
    for name in team_ids:
        if name not in rates:
            rates[name] = {"yellow_per_match": round(avg_y, 3),
                           "red_per_match": round(avg_r, 3), "matches": 0}

    payload = {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "competitions": competitions, "rates": rates}
    Path(cache).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return rates


def main():
    from data_loader import load_international_results, load_wc2026_fixtures
    key = get_api_key()
    if not key:
        print("Geen API-key gevonden. Maak 'apifootball_key.txt' aan met je sleutel.")
        print("Gratis account: https://www.api-football.com/")
        return
    df = load_international_results()
    wc = load_wc2026_fixtures(df)
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    print(f"{len(teams)} WK-landen. Team-ID's ophalen ...")
    ids = resolve_team_ids(key, teams)
    found = sum(1 for v in ids.values() if v)
    print(f"  {found}/{len(teams)} team-ID's gevonden.")
    print("Kaart-tarieven ophalen ...")
    rates = fetch_card_rates(key, ids)
    with_data = sum(1 for v in rates.values() if v["matches"] > 0)
    print(f"  {with_data} landen met echte kaartdata, rest op gemiddelde.")
    print("Opgeslagen in cards_cache.json.")


if __name__ == "__main__":
    main()
