"""
odds_fetcher.py
---------------
Haalt bookmaker-odds op van The Odds API en bewaart ze lokaal in een JSON-cache.

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

KNOCKOUT_SPORT_KEYS = [
    "soccer_fifa_world_cup",
    "soccer_world_cup_2026",
    "soccer_fifa_world_cup_2026",
]

KNOCKOUT_START_DATE = "2026-06-27"

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


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _decimal_odds_to_probs(home_odd: float, draw_odd: float,
                            away_odd: float) -> tuple[float, float, float]:
    """Zet decimale odds om naar marge-vrije kansen."""
    raw = [1.0 / home_odd, 1.0 / draw_odd, 1.0 / away_odd]
    s = sum(raw)
    return raw[0] / s, raw[1] / s, raw[2] / s


def _extract_h2h(event: dict) -> dict | None:
    """Vind de h2h-markt bij de eerst beschikbare voorkeursbookmaker."""
    bms = {b["key"]: b for b in event.get("bookmakers", [])}
    order = PREFERRED_BOOKMAKERS + [k for k in bms if k not in PREFERRED_BOOKMAKERS]
    for key in order:
        bm = bms.get(key)
        if bm is None:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            home, away = event["home_team"], event["away_team"]
            if home in outcomes and away in outcomes and "Draw" in outcomes:
                return {"bookmaker": key,
                        "home_odd": outcomes[home],
                        "draw_odd": outcomes["Draw"],
                        "away_odd": outcomes[away]}
    return None


def _extract_totals(event: dict) -> dict | None:
    """Vind de totals-markt (over/under). Kiest de meest gebalanceerde lijn.
    Retourneert {line, p_over, p_under} (marge-vrij) of None."""
    bms = {b["key"]: b for b in event.get("bookmakers", [])}
    order = PREFERRED_BOOKMAKERS + [k for k in bms if k not in PREFERRED_BOOKMAKERS]
    for key in order:
        bm = bms.get(key)
        if bm is None:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "totals":
                continue
            lines: dict = {}
            for o in market.get("outcomes", []):
                pt = o.get("point")
                if pt is None:
                    continue
                lines.setdefault(pt, {})[o["name"].lower()] = o["price"]
            best = None
            for pt, v in lines.items():
                if "over" not in v or "under" not in v:
                    continue
                po_raw, pu_raw = 1.0 / v["over"], 1.0 / v["under"]
                s = po_raw + pu_raw
                p_over, p_under = po_raw / s, pu_raw / s
                dist = abs(p_over - 0.5)
                if best is None or dist < best[0]:
                    best = (dist, pt, p_over, p_under)
            if best:
                _, pt, p_over, p_under = best
                return {"line": pt,
                        "p_over": round(p_over, 4),
                        "p_under": round(p_under, 4)}
    return None


def _fetch_events_for_sport(api_key: str, sport_key: str, regions: str) -> list:
    """Haal alle events op voor één sport-sleutel. Levert [] terug bij 404/422."""
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": regions,
              "markets": "h2h,totals", "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code in (404, 422):
        return []
    if resp.status_code == 401:
        raise RuntimeError("API-key afgewezen — controleer je sleutel.")
    resp.raise_for_status()
    return resp.json()


# ── publieke functies ──────────────────────────────────────────────────────────

def implied_total_goals(line: float, p_under: float) -> float | None:
    """Leid het verwachte totaal-doelpunten af uit P(under line) via Poisson CDF."""
    from math import floor
    from scipy.stats import poisson
    from scipy.optimize import brentq
    k = floor(line)
    if not (0.0 < p_under < 1.0):
        return None
    try:
        return float(brentq(lambda lam: poisson.cdf(k, lam) - p_under, 0.05, 12.0))
    except ValueError:
        return None


def get_implied_total(odds_db: dict, home: str, away: str) -> float | None:
    """Verwacht totaal-doelpunten uit de over/under-markt voor deze wedstrijd, of None."""
    info = odds_db.get(f"{home}|{away}") or odds_db.get(f"{away}|{home}")
    if not info or info.get("total_line") is None or info.get("p_under") is None:
        return None
    return implied_total_goals(info["total_line"], info["p_under"])


def fetch_odds(api_key: str, regions: str = "eu,uk") -> dict:
    """
    Haalt alle WK-wedstrijd-odds op (groepsfase én knockout), inclusief
    de over/under (totals) markt voor totaal-doelpunten-kalibratie.

    Retourneert:
        { "Home|Away": {p_home, p_draw, p_away, bookmaker, commence_time,
                        round, total_line, p_over, p_under}, ... }
    """
    odds_db: dict = {}
    tried_keys: set = set()

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
            if key not in odds_db:
                tot = _extract_totals(event)
                odds_db[key] = {
                    "home": h, "away": a,
                    "p_home": round(p_home, 4),
                    "p_draw": round(p_draw, 4),
                    "p_away": round(p_away, 4),
                    "bookmaker": h2h["bookmaker"],
                    "commence_time": ct,
                    "round": rnd,
                    "total_line": tot["line"] if tot else None,
                    "p_over":     tot["p_over"] if tot else None,
                    "p_under":    tot["p_under"] if tot else None,
                }
        if any(v["round"] == "knockout" for v in odds_db.values()):
            break
    return odds_db


def split_odds_by_round(odds_db: dict) -> tuple[dict, dict]:
    """Splits odds_db in (groep_odds, knockout_odds)."""
    group = {k: v for k, v in odds_db.items() if v.get("round", "group") == "group"}
    knockout = {k: v for k, v in odds_db.items() if v.get("round") == "knockout"}
    return group, knockout


def save_cache(odds_db: dict, path: str = "odds_cache.json") -> None:
    payload = {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_matches": len(odds_db), "odds": odds_db}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def load_cache(path: str = "odds_cache.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("odds", {})


def cache_info(path: str = "odds_cache.json") -> str:
    p = Path(path)
    if not p.exists():
        return "(geen odds-cache aanwezig)"
    payload = json.loads(p.read_text(encoding="utf-8"))
    return (f"{payload.get('n_matches', 0)} wedstrijden, "
            f"opgehaald op {payload.get('fetched_at')}")


def get_odds_probs(odds_db: dict, home: str,
                   away: str) -> tuple[float, float, float] | None:
    """Vind odds voor een wedstrijd, ook met thuis/uit omgedraaid."""
    key = f"{home}|{away}"
    if key in odds_db:
        e = odds_db[key]
        return e["p_home"], e["p_draw"], e["p_away"]
    key_rev = f"{away}|{home}"
    if key_rev in odds_db:
        e = odds_db[key_rev]
        return e["p_away"], e["p_draw"], e["p_home"]
    return None


def blend(model_probs: tuple[float, float, float],
          odds_probs: tuple[float, float, float] | None,
          weight_odds: float = 0.8) -> tuple[float, float, float]:
    """Gewogen gemiddelde van model- en odds-kansen."""
    if odds_probs is None:
        return model_probs
    return tuple(weight_odds * o + (1 - weight_odds) * m
                 for o, m in zip(odds_probs, model_probs))


# ── outright (titelkansen) ─────────────────────────────────────────────────────

OUTRIGHT_NAME_MAP = {
    "United States": "United States", "USA": "United States", "US": "United States",
    "Korea Republic": "South Korea", "South Korea": "South Korea",
    "Czechia": "Czech Republic", "Czech Republic": "Czech Republic",
    "Ivory Coast": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo", "Congo DR": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey", "Turkey": "Turkey",
    "Cabo Verde": "Cape Verde", "Cape Verde": "Cape Verde",
    "Curacao": "Curaçao", "Curaçao": "Curaçao",
}


def fetch_outright_odds(api_key: str, regions: str = "eu,uk") -> dict[str, float]:
    """Haal WK 2026 outright-winnaar-kansen op via de Odds API."""
    from collections import defaultdict
    base = ODDS_API_BASE

    sports_resp = requests.get(f"{base}/sports/",
                               params={"apiKey": api_key, "all": "true"}, timeout=30)
    sports_resp.raise_for_status()
    all_sports = sports_resp.json()

    candidates = [s["key"] for s in all_sports
                  if any(k in s["key"].lower() for k in
                         ["world_cup_winner", "world_cup_2026", "fifa_world_cup_winner"])
                  or ("world_cup" in s.get("title", "").lower()
                      and "winner" in s.get("title", "").lower())]
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
                    continue
                resp.raise_for_status()
                for event in resp.json():
                    for bm in event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            for outcome in mkt.get("outcomes", []):
                                name = outcome.get("name", "")
                                price = outcome.get("price", 0)
                                if price > 1 and name not in ("Field", "The Field"):
                                    raw[name].append(1.0 / price)
                if raw:
                    break
            except Exception as e:
                last_error = e
        if raw:
            break

    if not raw:
        raise ValueError(f"Geen outright-kansen gevonden. Laatste fout: {last_error}.")

    avg = {t: sum(p) / len(p) for t, p in raw.items()}
    total = sum(avg.values())
    normalized = {t: p / total for t, p in avg.items()}
    result = {OUTRIGHT_NAME_MAP.get(n, n): p for n, p in normalized.items()}
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
