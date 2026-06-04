# WK 2026 poule-voorspeller

Een compact, leesbaar Python-skelet om de uitslagen van het WK voetbal 2026 te
voorspellen — gericht op het **winnen van een werkpoule**. Eenvoud boven alles:
**Elo-ratings → Poisson-scoremodel → Monte-Carlo-simulatie**, met twee
verfijningen die er aantoonbaar toe doen: **hoogte-correctie** voor Mexicaanse
venues en **blend met bookmaker-odds** die je vlak voor de aftrap kunt
verversen.

## Wat het doet

1. **Data** — laadt ~49.000 internationale wedstrijden (1872–heden) plus de echte
   WK 2026-fixtures, rechtstreeks van een gratis bron. Geen account nodig.
2. **Elo** — berekent de sterkte van elk land uit de volledige historie.
3. **Scoremodel** — zet twee ratings om naar verwachte doelpunten en een volledige
   kansverdeling over uitslagen (Poisson + Dixon-Coles-correctie).
4. **Hoogte-correctie** — Mexicaanse venues (Mexico City 2240m, Zapopan 1566m)
   penalty voor zeespiegel-teams, bonus voor geacclimatiseerde teams (Mexico,
   Colombia, Ecuador).
5. **Odds-blend** — leest een lokale cache met de meest recente bookmaker-odds
   en mengt die met het model (80% odds / 20% model). Closing odds zijn het
   scherpst omdat ze late blessures inprijzen — daarom verversbaar.
6. **Simulatie** — speelt het hele toernooi tienduizenden keren na en schat zo
   de kans dat elk land elke fase haalt.

## Snel starten

```bash
pip install -r requirements.txt
python run_demo.py
```

Dat werkt direct, zonder API-key — alleen op basis van het model. Dataset wordt
automatisch opgehaald en lokaal gecached.

## Odds toevoegen (sterk aanbevolen)

1. Maak een gratis account aan op **https://the-odds-api.com/** (500 calls/maand
   gratis — meer dan genoeg).
2. Sla je sleutel op. Twee opties:
   - **Makkelijkst:** maak een bestand `odds_api_key.txt` in deze map en zet
     daar alleen je sleutel in (geen aanhalingstekens, geen `key=`).
   - Of via een omgevingsvariabele: `setx ODDS_API_KEY "jouw_sleutel"` in
     PowerShell (eenmalig).
3. Ververs de odds vlak voor een speeldag:
   ```bash
   python refresh_odds.py
   ```
4. Draai het model — de odds worden nu automatisch geblend:
   ```bash
   python run_demo.py
   ```

In de output zie je een `+` achter wedstrijden waarvoor odds beschikbaar waren.
Verversen kun je zo vaak als je wilt; de cache wordt elke keer overschreven.

## Bestanden

| Bestand            | Rol                                                                |
|--------------------|--------------------------------------------------------------------|
| `data_loader.py`   | data inladen (GitHub-bron + Kaggle- en Odds-API-routes)            |
| `elo.py`           | Elo-ratings berekenen uit de historie                              |
| `altitude.py`      | hoogte-correctie voor Mexicaanse venues                            |
| `model.py`         | Elo → verwachte goals → Poisson-matrix → 1X2-kansen + blend        |
| `odds_fetcher.py`  | odds ophalen van The Odds API en lokaal cachen                     |
| `refresh_odds.py`  | los script om de odds-cache te verversen                           |
| `simulate.py`      | Monte-Carlo-simulatie van groepsfase + knock-out                   |
| `run_demo.py`      | koppelt alles aan elkaar en print de voorspellingen                |

## Poulestrategie

Het model geeft **kansen**. Hoe je die invult hangt af van de scoreregels:

- **Punten voor exacte uitslag**: vul bij favorieten de kolom `uitslag` in
  (meestal `1-0` of `2-0`) — wiskundig optimaal.
- **Punten voor tendens (1X2)**: kies de hoogste van de drie kansen.
- **Knock-out / kampioen**: gebruik de simulatiekansen.

## Bekende beperkingen

- **Knock-out-bracket is een benadering.** `simulate.build_bracket` seedt teams
  op Elo in een vast schema. Vervang door het officiële WK-schema zodra dat
  vaststaat. De groepsvoorspellingen en de onderlinge volgorde lijden hier niet
  onder.
- **Odds-blend zit in de wedstrijdvoorspellingen, niet in de simulatie.** De
  Monte-Carlo gebruikt alleen Elo+hoogte. Voor de simulatie levert dat geen
  meerwaarde omdat odds vooral kalibratie verbeteren, niet rangordening.
- **Hoogte-effect is generiek** — alle teams behalve Mexico/Colombia/Ecuador
  krijgen dezelfde penalty. Je kunt fijnregelen via `ACCLIMATIZED_TEAMS` in
  `altitude.py`.
- **Reistijden en spelersvermoeidheid** zijn bewust niet ingebouwd: het
  empirisch bewijs dat het op teamniveau echt uitmaakt is dun.

## Wat verversen versus alles opnieuw draaien?

- `python refresh_odds.py` — alleen odds opnieuw ophalen (~1 seconde, 1 API-call)
- `python run_demo.py` — model + simulatie opnieuw draaien (~30 seconden)

Best practice: ververs de odds vlak voor een speeldag en draai dan `run_demo.py`
nog één keer.
