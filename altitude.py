"""
altitude.py
-----------
Hoogte-correctie voor het WK 2026.

Onderzoek van FIFA (WK 2010) toont dat teams op hoogte ~3% minder afstand
afleggen. Op de hoogte van Mexico-Stad (2240m) is het zuurstofgehalte ~25%
lager. Teams die structureel op hoogte spelen (Mexico, Colombia, Ecuador)
zijn geacclimatiseerd; Europese ploegen niet.

We modelleren dit als een tijdelijke Elo-aanpassing per wedstrijd:
  - onder 1200m: geen effect (drempel uit de literatuur)
  - boven 1200m: lineair oplopende penalty voor niet-geacclimatiseerde teams
  - geacclimatiseerde teams: kleine BONUS (tegenstander is verzwakt)
"""

from __future__ import annotations

# Hoogte in meters per WK 2026-speelstad. De dataset gebruikt de exacte
# gemeente-naam — niet de bekende plaatsnaam. Zo ligt Estadio Akron in Zapopan
# (buitenwijk Guadalajara), en Estadio BBVA in Guadalupe (omgeving Monterrey).
VENUE_ALTITUDE = {
    "Mexico City": 2240,    # Estadio Azteca / Banorte
    "Zapopan": 1566,        # Estadio Akron (Guadalajara-stadion)
    "Guadalupe": 540,       # Estadio BBVA (Monterrey-stadion) — onder de drempel
    "Guadalajara": 1566,    # als alias, voor het geval de dataset wisselt
    "Monterrey": 540,       # idem alias
    # alle US- en Canadese steden liggen praktisch op zeeniveau (default 0)
}

# Teams die structureel op hoogte spelen. Mexico speelt thuis in Mexico City,
# Colombia in Bogotá (2640m), Ecuador in Quito (2850m).
ACCLIMATIZED_TEAMS = {"Mexico", "Colombia", "Ecuador"}

ALTITUDE_THRESHOLD = 1200       # onder deze hoogte: geen effect
MAX_PENALTY_ELO = 60            # max penalty voor niet-geacclimatiseerde teams
MAX_BONUS_ELO = 20              # max bonus voor geacclimatiseerde teams
SATURATION_ALTITUDE = 2200      # boven deze hoogte: max effect


def altitude_adjustment(team: str, city: str) -> float:
    """
    Tijdelijke Elo-correctie voor deze specifieke wedstrijd op dit venue.
    Wordt opgeteld bij de team-Elo voordat we lambda berekenen.
    """
    altitude = VENUE_ALTITUDE.get(city, 0)
    if altitude < ALTITUDE_THRESHOLD:
        return 0.0
    # severity loopt lineair van 0 (op 1200m) tot 1 (op 2200m+)
    span = SATURATION_ALTITUDE - ALTITUDE_THRESHOLD
    severity = min(1.0, (altitude - ALTITUDE_THRESHOLD) / span)
    if team in ACCLIMATIZED_TEAMS:
        return +MAX_BONUS_ELO * severity
    return -MAX_PENALTY_ELO * severity


if __name__ == "__main__":
    # snelle sanity check
    cases = [
        ("Netherlands", "Mexico City"),
        ("Mexico", "Mexico City"),
        ("Colombia", "Mexico City"),
        ("Netherlands", "Guadalajara"),
        ("Netherlands", "Atlanta"),
        ("Netherlands", "Monterrey"),
    ]
    for team, city in cases:
        print(f"  {team:15s} in {city:15s} -> Elo {altitude_adjustment(team, city):+.1f}")
