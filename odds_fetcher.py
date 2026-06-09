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

# Voorkeursvolgorde voor bookmakers: scherpste eerst.
# Pinnacle = lage marge, hoge limieten -> meest informatieve odds.
PREFERRED_BOOKMAKERS = ["pinnacle", "bet365", "betfair", "williamhill", "unibet"]

# Sommige teamnamen verschillen tussen The Odds API en onze Kaggle-dataset.
# Vul deze tabel aan als je verschillen tegenkomt (linkerkant = Odds API, rechts = onze data).
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


def fetch_odds(api_key: str, regions: str = "eu,uk") -> dict:
    """
    Haalt alle WK-wedstrijd-odds op. Retourneert een dict:
        { "Home_Team|Away_Team": {p_home, p_draw, p_away, bookmaker, date}, ... }
    """
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError("API-key afgewezen — controleer je sleutel.")
    resp.raise_for_status()
    events = resp.json()

    odds_db: dict = {}
    for event in events:
        h = _normalize_team(event["home_team"])
        a = _normalize_team(event["away_team"])
        h2h = _extract_h2h(event)
        if h2h is None:
            continue
        p_home, p_draw, p_away = _decimal_odds_to_probs(h2h["home_odd"], h2h["draw_odd"], h2h["away_odd"])
        odds_db[f"{h}|{a}"] = {
            "home": h, "away": a,
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "bookmaker": h2h["bookmaker"],
            "commence_time": event.get("commence_time"),
        }
    return odds_db


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


def fetch_outright_odds(api_key: str, sport_key: str = "soccer_fifa_world_cup",
                        regions: str = "eu,uk") -> dict[str, float]:
    """
    Haal de outright winner-markt op (titelkansen per team) via de Odds API.
    Retourneert {dataset_teamnaam: genormaliseerde_implied_probability}.

    De ruwe implied kansen (1/decimale_odds) worden genormaliseerd om de
    bookmaker-overround te verwijderen, zodat ze optellen tot 1.0.
    """
    import requests
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": api_key, "regions": regions,
              "markets": "outrights", "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # verzamel implied kansen per team over alle bookmakers
    from collections import defaultdict
    raw: dict[str, list[float]] = defaultdict(list)
    for event in data:
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") == "outrights":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if price > 1:
                            raw[name].append(1.0 / price)

    if not raw:
        return {}

    # gemiddelde per team (over bookmakers)
    avg = {t: sum(p) / len(p) for t, p in raw.items()}

    # normaliseer: verwijder de overround
    total = sum(avg.values())
    normalized = {t: p / total for t, p in avg.items()}

    # vertaal naar onze dataset-namen
    result: dict[str, float] = {}
    for api_name, prob in normalized.items():
        mapped = OUTRIGHT_NAME_MAP.get(api_name, api_name)
        result[mapped] = prob

    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
