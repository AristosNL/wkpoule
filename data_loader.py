"""
data_loader.py
--------------
Inladen van data uit drie bronnen:

1. Internationale resultaten 1872-heden  -> ECHTE bron, werkt direct (GitHub-mirror
   van de bekende Kaggle-dataset martj42). Bevat ook de WK 2026-fixtures.
2. Kaggle                                -> alternatieve route via de officiele Kaggle-API
                                            (vereist account + kaggle.json). Optioneel.
3. The Odds API                          -> live/recente bookmaker-odds (vereist gratis
                                            API-key). Optioneel, alleen nodig als je odds
                                            wilt mengen met het Elo-model.

Voor de poule heb je in principe alleen bron 1 nodig.
"""

from __future__ import annotations
import io
import os
import requests
import pandas as pd

# De martj42-dataset wordt dagelijks bijgewerkt en staat ook op GitHub als losse CSV.
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def load_international_results(url: str = RESULTS_URL, cache_path: str = "results.csv") -> pd.DataFrame:
    """Laad alle internationale wedstrijden. Cachet lokaal zodat je niet elke keer downloadt."""
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(cache_path, index=False)

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_played_and_fixtures(df: pd.DataFrame):
    """Splits in (a) gespeelde wedstrijden met scores en (b) toekomstige fixtures (score = NA)."""
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    fixtures = df[df["home_score"].isna()].copy()
    return played, fixtures


def load_wc2026_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Haal alleen de WK 2026-wedstrijden eruit (groepsfase zit als fixture in de data)."""
    mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
    return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# OPTIONEEL: Kaggle-route (alleen als je liever de officiele Kaggle-API gebruikt)
# ---------------------------------------------------------------------------
def load_from_kaggle(dataset: str = "martj42/international-football-results-from-1872-to-2017",
                     filename: str = "results.csv") -> pd.DataFrame:
    """
    Vereist: `pip install kaggle` en een ~/.kaggle/kaggle.json met je API-token
    (Account -> Create New API Token op kaggle.com). Daarna:

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files(dataset, path=".", unzip=True)
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=".", unzip=True)
    return pd.read_csv(filename)


# ---------------------------------------------------------------------------
# OPTIONEEL: The Odds API (live odds, gratis tier op the-odds-api.com)
# ---------------------------------------------------------------------------
def fetch_odds(api_key: str, sport_key: str = "soccer_fifa_world_cup",
               regions: str = "eu", markets: str = "h2h") -> list[dict]:
    """
    Haalt actuele 1X2-odds (h2h) op voor het WK. Zet je key in de omgeving:
        export ODDS_API_KEY=...
    Gratis tier = 500 requests/maand, ruim genoeg voor een poule.
    Retourneert de ruwe JSON-lijst; gebruik odds_to_probs() om er kansen van te maken.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def odds_to_probs(home_odd: float, draw_odd: float, away_odd: float) -> tuple[float, float, float]:
    """
    Zet decimale odds om naar kansen en haalt de bookmaker-marge eruit (normaliseren).
    1/odd is de impliciete kans inclusief marge; delen door de som corrigeert daarvoor.
    """
    raw = [1 / home_odd, 1 / draw_odd, 1 / away_odd]
    s = sum(raw)
    return tuple(p / s for p in raw)  # (P_thuis, P_gelijk, P_uit)


if __name__ == "__main__":
    df = load_international_results()
    played, fixtures = split_played_and_fixtures(df)
    wc = load_wc2026_fixtures(df)
    print(f"Gespeeld: {len(played)} | Toekomstige fixtures: {len(fixtures)}")
    print(f"WK 2026-wedstrijden in data: {len(wc)}")
