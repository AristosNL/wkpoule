"""
odds_fetcher.py
---------------
Haalt bookmaker-odds op van The Odds API en bewaart ze lokaal in een JSON-cache.

Belangrijk: 'closing odds' (vlak voor de aftrap) zijn aantoonbaar het scherpst
omdat ze late blessures en opstellingsnieuws inprijzen. Daarom kun je dit
script zo vaak draaien als je wilt; elke ronde overschrijft de cache.

Gebruik:
    python refresh_odds.py             # ververst de cache
    -> odds_cache.json wordt bijgewerkt
"""

from __future__ import annotations
import json
import os
import time
import requests
from pathlib import Path

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

# Alternatieve sport-sleutels die de Odds API soms gebruikt voor knockout-ronden.
# De API probeert eerst de primaire sleutel; als die geen knockout-wedstrijden geeft,
# worden deze ook geprobeerd.
KNOCKOUT_SPORT_KEYS = [
    "soccer_fifa_world_cup",
    "soccer_world_cup_2026",
    "soccer_fifa_world_cup_2026",
]

# Groepsfase-datumgrens: wedstrijden na deze datum zijn knockout-rondes.
# WK 2026: groepsfase eindigt 26 juni; knockout start 28 juni.
KNOCKOUT_START_DATE = "2026-06-27"

# Voorkeursvolgorde voor bookmakers: scherpste eerst.
PREFERRED_BOOKMAKERS = ["pinnacle", "bet365", "betfair", "williamhill", "unibet"]

TEAM_NAME_MAP = {
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def _normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _decimal_odds_to_probs(home_odd: float, draw_odd: float, away_odd: float) -> tuple[float, float, float]:
    """Zet decimale odds om naar marge-vrije kansen (overround eruit halen)."""
    raw = [1.0 / home_odd, 1.0 / draw_odd, 1.0 / away_odd]
    s = sum(raw)
    return raw[0] / s, raw[1] / s, raw[2] / s


def _extract_h2h(event: dict) -> dict | None:
    """Vind de h2h-markt bij de eerst beschikbare voorkeursbookmaker."""
    bms = {b["key"]: b for b in event.get("bookmakers", [])}
    for pref in PREFERRED_BOOKMAKERS:
        if pref not in bms:
            continue
        for market in bms[pref].get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            home, away = event["home_team"], event["away_team"]
            if home in outcomes and away in outcomes and "Draw" in outcomes:
                return {
                    "bookmaker": pref,
                    "home_odd": outcomes[home],
                    "draw_odd": outcomes["Draw"],
                    "away_odd": outcomes[away],
                }
    # geen voorkeursbookmaker beschikbaar -> pak de eerste die werkt
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            home, away = event["home_team"], event["away_team"]
            if home in outcomes and away in outcomes and "Draw" in outcomes:
                return {
                    "bookmaker": bm["key"],
                    "home_odd": outcomes[home],
                    "draw_odd": outcomes["Draw"],
                    "away_odd": outcomes[away],
                }
    return None


def _fetch_events_for_sport(api_key: str, sport_key: str, regions: str) -> list:
    """Haal alle events op voor één sport-sleutel. Levert [] terug bij 404/422."""
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": "h2h",
              "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code in (404, 422):
        return []
    if resp.status_code == 401:
        raise RuntimeError("API-key afgewezen — controleer je sleutel.")
    resp.raise_for_status()
    return resp.json()


def fetch_odds(api_key: str, regions: str = "eu,uk") -> dict:
    """
    Haalt alle WK-wedstrijd-odds op (groepsfase én knockout).

    Retourneert:
        { "Home|Away": {p_home, p_draw, p_away, bookmaker, commence_time, round}, ... }

    'round' is 'group' voor groepswedstrijden en 'knockout' voor knock-out-duels,
    bepaald op basis van KNOCKOUT_START_DATE.

    Strategie: probeer KNOCKOUT_SPORT_KEYS achtereenvolgens. Verifieer voor elke sleutel
    of er knockout-wedstrijden in zitten; stop zodra we een volledige set hebben.
    """
    odds_db: dict = {}
    tried_keys = set()

    for sport_key in KNOCKOUT_SPORT_KEYS:
        if sport_key in tried_keys:
            continue
        tried_keys.add(sport_key)
        events = _fetch_events_for_sport(api_key, sport_key, regions)
        for event in events:
            h = _normalize_team(event["home_team"])
            a = _normalize_team(event["away_team"])
            h2h = _extract_h2h(event)
            if h2h is None:
                continue
            p_home, p_draw, p_away = _decimal_odds_to_probs(
                h2h["home_odd"], h2h["draw_odd"], h2h["away_odd"])
            ct = event.get("commence_time", "")
            rnd = "knockout" if ct >= KNOCKOUT_START_DATE else "group"
            key = f"{h}|{a}"
            if key not in odds_db:   # eerste (scherpste) bookmaker wint
                odds_db[key] = {
                    "home": h, "away": a,
                    "p_home": round(p_home, 4),
                    "p_draw": round(p_draw, 4),
                    "p_away": round(p_away, 4),
                    "bookmaker": h2h["bookmaker"],
                    "commence_time": ct,
                    "round": rnd,
                }
        # als we knockout-wedstrijden hebben gevonden, zijn we klaar
        if any(v["round"] == "knockout" for v in odds_db.values()):
            break
    return odds_db


def split_odds_by_round(odds_db: dict) -> tuple[dict, dict]:
    """Splits odds_db in (groep_odds, knockout_odds) op basis van het 'round'-veld."""
    group = {k: v for k, v in odds_db.items() if v.get("round", "group") == "group"}
    knockout = {k: v for k, v in odds_db.items() if v.get("round") == "knockout"}
    return group, knockout


def save_cache(odds_db: dict, path: str = "odds_cache.json") -> None:
    payload = {
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_matches": len(odds_db),
        "odds": odds_db,
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cache(path: str = "odds_cache.json") -> dict:
    """Laad de odds-cache. Retourneert lege dict als die er nog niet is."""
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload.get("odds", {})


def cache_info(path: str = "odds_cache.json") -> str:
    p = Path(path)
    if not p.exists():
        return "(geen odds-cache aanwezig)"
    payload = json.loads(p.read_text(encoding="utf-8"))
    return f"{payload.get('n_matches', 0)} wedstrijden, opgehaald op {payload.get('fetched_at')}"


def get_odds_probs(odds_db: dict, home: str, away: str) -> tuple[float, float, float] | None:
    """Probeer odds te vinden voor een wedstrijd, ook met thuis/uit omgedraaid."""
    key = f"{home}|{away}"
    if key in odds_db:
        e = odds_db[key]
        return e["p_home"], e["p_draw"], e["p_away"]
    # API kan de wedstrijd ook andersom hebben opgeslagen
    key_rev = f"{away}|{home}"
    if key_rev in odds_db:
        e = odds_db[key_rev]
        return e["p_away"], e["p_draw"], e["p_home"]
    return None


def blend(model_probs: tuple[float, float, float],
          odds_probs: tuple[float, float, float] | None,
          weight_odds: float = 0.8) -> tuple[float, float, float]:
    """
    Gewogen gemiddelde van model- en odds-kansen. Als odds ontbreken, val terug
    op het model. weight_odds = 0.8 betekent: 80% gewicht aan de markt, 20% aan
    het Elo-model.
    """
    if odds_probs is None:
        return model_probs
    return tuple(weight_odds * o + (1 - weight_odds) * m
                 for o, m in zip(odds_probs, model_probs))


# Namen zoals de Odds API ze geeft -> onze dataset-naam
OUTRIGHT_NAME_MAP = {
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Republic of Ireland": "Ireland",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Cape Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
}


def fetch_outright_odds(api_key: str, regions: str = "eu,uk") -> dict[str, float]:
    """
    Haal de WK 2026 outright winner-kansen op via de Odds API.

    Strategie:
      1. Vraag /v4/sports/?all=true op (geen credits) om alle beschikbare
         sport-sleutels te vinden die lijken op een WK-winnaarsmarkt.
      2. Haal van elke kandidaat-sleutel de odds op (h2h of outrights).
      3. Normaliseer de implied kansen over alle bookmakers.
    """
    import requests
    from collections import defaultdict

    base = "https://api.the-odds-api.com/v4"

    # stap 1: vind de juiste sport-sleutel (gratis, kost geen credits)
    sports_resp = requests.get(f"{base}/sports/",
                               params={"apiKey": api_key, "all": "true"}, timeout=30)
    sports_resp.raise_for_status()
    all_sports = sports_resp.json()

    # zoek sport-sleutels die betrekking hebben op WK 2026 winnaar
    candidates = [s["key"] for s in all_sports
                  if any(k in s["key"].lower() for k in
                         ["world_cup_winner", "world_cup_2026", "fifa_world_cup_winner"])
                  or ("world_cup" in s.get("title", "").lower() and
                      "winner" in s.get("title", "").lower())]

    # als er geen gerichte treffer is: val terug op de gewone soccer_fifa_world_cup
    # en probeer markets=outrights — sommige sleutels ondersteunen het toch
    if not candidates:
        candidates = ["soccer_fifa_world_cup"]

    raw: dict[str, list[float]] = defaultdict(list)
    last_error = None
    for sport_key in candidates:
        for market in ("h2h", "outrights"):
            try:
                resp = requests.get(f"{base}/sports/{sport_key}/odds/",
                                    params={"apiKey": api_key, "regions": regions,
                                            "markets": market, "oddsFormat": "decimal"},
                                    timeout=30)
                if resp.status_code == 422:
                    continue        # markttype niet ondersteund, probeer de andere
                resp.raise_for_status()
                data = resp.json()
                for event in data:
                    for bm in event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            for outcome in mkt.get("outcomes", []):
                                name = outcome.get("name", "")
                                price = outcome.get("price", 0)
                                if price > 1 and name not in ("Field", "The Field"):
                                    raw[name].append(1.0 / price)
                if raw:
                    break   # gegevens gevonden, stop met zoeken
            except Exception as e:
                last_error = e
        if raw:
            break

    if not raw:
        raise ValueError(f"Geen outright-kansen gevonden. Laatste fout: {last_error}. "
                         f"Beschikbare WK-sleutels: {candidates}")

    # gemiddelde over bookmakers + overround verwijderen
    avg = {t: sum(p) / len(p) for t, p in raw.items()}
    total = sum(avg.values())
    normalized = {t: p / total for t, p in avg.items()}

    # vertaal naar onze dataset-namen
    result: dict[str, float] = {}
    for api_name, prob in normalized.items():
        mapped = OUTRIGHT_NAME_MAP.get(api_name, api_name)
        result[mapped] = prob

    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
