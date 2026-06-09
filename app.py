"""
app.py — Streamlit-app voor de WK 2026 poule-voorspeller
=========================================================
Een visuele schil rond je bestaande model. Niets aan de modellogica verandert;
deze app roept dezelfde functies aan als run_demo.py, maar dan in de browser.

Starten (eenmalig per sessie, in PowerShell, vanuit je projectmap):
    streamlit run app.py

Daarna opent je browser vanzelf. De terminal mag je verder negeren.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from data_loader import (load_international_results, split_played_and_fixtures,
                         load_wc2026_fixtures)
from elo import compute_elo
from model import calibrate, predict_match
from simulate import derive_groups, simulate_tournament
from odds_fetcher import (load_cache, cache_info, fetch_odds, save_cache,
                          fetch_outright_odds)
from poule_strategy import ScoringRules, optimal_prediction, expected_points
from poule_extras import (simulate_full, recommend_group_standing,
                          recommend_qualifiers, recommend_goal_leaders,
                          STAGE_POINTS_DEFAULT, STAGE_COUNTS)
from cards_model import (load_card_rates, simulate_cards,
                         recommend_most_cards, recommend_fewest_cards)
from cards_fetcher import resolve_team_ids, fetch_card_rates


def get_secret(name: str, fallback_file: str | None = None) -> str | None:
    """
    Lees een geheime sleutel uit (in volgorde):
      1. st.secrets         -> op Streamlit Cloud
      2. omgevingsvariabele -> lokaal als je 'm zo hebt gezet
      3. tekstbestand       -> lokaal als je 'm in een .txt hebt staan
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass  # geen secrets.toml lokaal — prima
    val = os.environ.get(name)
    if val:
        return val.strip()
    if fallback_file and Path(fallback_file).exists():
        return Path(fallback_file).read_text(encoding="utf-8").strip()
    return None


st.set_page_config(page_title="WK 2026 Poule-voorspeller", page_icon="⚽", layout="wide")


# ----------------------------------------------------------------------------
# Dure stappen cachen: data laden + Elo + kalibratie gebeurt maar één keer.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Data laden en Elo-ratings berekenen ...")
def load_everything():
    df = load_international_results()
    played, _ = split_played_and_fixtures(df)
    wc = load_wc2026_fixtures(df)
    ratings, history = compute_elo(played)
    cal = calibrate(history)
    # wc als records teruggeven zodat het cachebaar/serializeerbaar is
    return ratings, cal, wc.to_dict("records")


def run_simulation(ratings, cal, wc_records, n_sims, odds_db, weight_odds):
    """Draai de simulatie met de odds en het gewicht zoals nu in de zijbalk staan.
    Resultaat in session_state zodat het tussen reruns/tabs gedeeld wordt."""
    wc_local = pd.DataFrame(wc_records)
    groups, matches = derive_groups(wc_local)
    with st.spinner(f"Simuleren ({n_sims:,} runs"
                    + (f", odds-blend {weight_odds*100:.0f}% over {len(odds_db)} wedstrijden"
                       if odds_db and weight_odds > 0 else ", puur model")
                    + ") ..."):
        sim = simulate_full(ratings, cal, groups, matches, n_sims=n_sims,
                            odds_db=odds_db or None, weight_odds=weight_odds)
    st.session_state["sim_result"] = sim
    st.session_state["sim_meta"] = {
        "n_sims": n_sims,
        "with_odds": bool(odds_db and weight_odds > 0),
        "n_odds": len(odds_db) if odds_db else 0,
        "weight": weight_odds,
    }


# ----------------------------------------------------------------------------
# Heatmap-helpers (plotly)
# ----------------------------------------------------------------------------
def probability_heatmap(matrix, home, away, n=6):
    z = matrix[:n, :n] * 100
    text = [[f"{z[i, j]:.1f}%" for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(j) for j in range(n)], y=[str(i) for i in range(n)],
        text=text, texttemplate="%{text}", textfont={"size": 12},
        colorscale="Blues", showscale=True, hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title=f"{away} — goals", yaxis_title=f"{home} — goals",
        yaxis=dict(autorange="reversed"), height=430,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def expected_points_heatmap(matrix, rules, home, away, n=6):
    z = np.zeros((n, n))
    for h in range(n):
        for a in range(n):
            z[h, a] = expected_points(h, a, matrix, rules)
    text = [[f"{z[i, j]:.2f}" for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(j) for j in range(n)], y=[str(i) for i in range(n)],
        text=text, texttemplate="%{text}", textfont={"size": 12},
        colorscale="Tealgrn", showscale=True, hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title=f"{away} — voorspelling", yaxis_title=f"{home} — voorspelling",
        yaxis=dict(autorange="reversed"), height=430,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def render_bracket_html(bracket_data: list) -> str:
    """Bouw een HTML-bracket uit de modale teams per slot. 5 kolommen + winnaar."""
    short = {
        "Bosnia and Herzegovina": "Bosnia & H.", "Czech Republic": "Czech R.",
        "South Korea": "S. Korea", "United States": "USA", "DR Congo": "DR Congo",
        "Saudi Arabia": "Saudi A.", "Ivory Coast": "Ivory C.", "New Zealand": "N. Zealand",
        "Cape Verde": "Cape V.",
    }
    sh = lambda t: short.get(t, t)
    titles = {"R32": "Laatste 32", "R16": "Achtste finale", "QF": "Kwartfinale",
              "SF": "Halve finale", "F": "Finale"}

    def cell(name, prob, is_winner):
        # gedempt tonen als de positie onzeker is (<25%)
        muted = "" if prob >= 0.25 else "opacity:0.65;font-style:italic;"
        cls = "team winner" if is_winner else "team"
        return (f'<div class="{cls}" style="{muted}">'
                f'<span class="n">{sh(name)}</span>'
                f'<span class="p">{prob*100:.0f}%</span></div>')

    cols_html = []
    champion = None
    for r in bracket_data:
        rn = r["round"]
        items = [f'<div class="round-title">{titles.get(rn, rn)}</div>']
        for m in r["matches"]:
            w = m["winner"]
            home_winner = (m["home"] == w)
            away_winner = (m["away"] == w)
            items.append(
                '<div class="match">'
                + cell(m["home"], m["p_home"], home_winner)
                + cell(m["away"], m["p_away"], away_winner)
                + '</div>')
        cols_html.append('<div class="round">' + "".join(items) + '</div>')
        if rn == "F" and r["matches"]:
            champion = r["matches"][0]

    # extra kolom voor de wereldkampioen
    if champion:
        w_html = (
            '<div class="round">'
            '<div class="round-title">Wereldkampioen</div>'
            '<div class="champion-wrap">'
            f'<div class="champion-box">🏆<br>{sh(champion["winner"])}'
            f'<div class="champion-prob">{champion["p_winner"]*100:.0f}%</div></div></div></div>'
        )
        cols_html.append(w_html)

    css = """
    <style>
      body { font-family: -apple-system, system-ui, sans-serif; margin: 0; }
      .bracket { display: flex; gap: 6px; padding: 6px; min-height: 1100px; font-size: 11px; }
      .round { flex: 1; display: flex; flex-direction: column; justify-content: space-around; min-width: 0; }
      .round-title { background: #185FA5; color: white; padding: 5px 6px; text-align: center;
                     font-weight: 600; border-radius: 4px; margin-bottom: 4px; font-size: 11px;
                     letter-spacing: 0.02em; }
      .match { background: #F8F8F6; border: 1px solid #E0E0D8; border-radius: 4px;
               padding: 3px 4px; margin: 2px 0; }
      .team { display: flex; justify-content: space-between; align-items: center;
              padding: 2px 4px; gap: 6px; }
      .team.winner { background: #D0E5FA; font-weight: 600; border-radius: 3px; color: #042C53; }
      .team .n { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .team .p { font-size: 9px; color: #999; }
      .team.winner .p { color: #185FA5; }
      .champion-wrap { display: flex; flex-direction: column; justify-content: center; flex: 1; }
      .champion-box { background: linear-gradient(135deg, #FFD466, #E89B1A); color: #3A2400;
                      text-align: center; padding: 18px 8px; border-radius: 8px;
                      font-weight: 700; font-size: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
      .champion-prob { font-size: 11px; font-weight: 500; margin-top: 6px; color: #5A3A00; }
    </style>
    """
    return css + '<div class="bracket">' + "".join(cols_html) + '</div>'


# ----------------------------------------------------------------------------
# Sidebar — instellingen
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Instellingen")


def get_secret(name: str) -> str | None:
    """Probeer st.secrets, dan omgevingsvariabele, dan lokaal bestand. Voor cloud + lokaal."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    val = os.environ.get(name)
    if val:
        return val.strip()
    local_files = {"ODDS_API_KEY": "odds_api_key.txt",
                   "APIFOOTBALL_KEY": "apifootball_key.txt"}
    fn = local_files.get(name)
    if fn and Path(fn).exists():
        return Path(fn).read_text(encoding="utf-8").strip()
    return None


# initialiseer odds in sessie (overleeft niet een server-herstart op de cloud)
if "odds_db" not in st.session_state:
    st.session_state["odds_db"] = load_cache()   # leeg bij eerste run in de cloud

st.sidebar.subheader("Bookmaker-odds")
_odds_loaded = len(st.session_state["odds_db"])
st.sidebar.caption(f"{_odds_loaded} wedstrijden in cache" if _odds_loaded
                   else "(nog geen odds opgehaald)")

api_key = get_secret("ODDS_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("API-key (the-odds-api.com)", value="",
                                    type="password",
                                    help="Op de cloud: zet ODDS_API_KEY in Secrets. "
                                         "Lokaal: in odds_api_key.txt of als omgevingsvariabele.")

if st.sidebar.button("🔄 Ververs odds nu"):
    if not api_key:
        st.sidebar.error("Geen API-key beschikbaar.")
    else:
        try:
            odds = fetch_odds(api_key)
            if odds:
                st.session_state["odds_db"] = odds
                try:
                    save_cache(odds)
                except Exception:
                    pass
                st.sidebar.success(f"{len(odds)} wedstrijden opgehaald.")
            else:
                st.sidebar.warning("Geen wedstrijden teruggekregen "
                                   "(WK-odds mogelijk nog niet gepubliceerd).")
        except Exception as e:
            st.sidebar.error(f"Fout bij ophalen: {e}")

st.sidebar.subheader("Bookmaker-titelkansen")
if "outright_probs" not in st.session_state:
    st.session_state["outright_probs"] = {}
_n_out = len(st.session_state["outright_probs"])
st.sidebar.caption(f"{_n_out} landen met titelkansen" if _n_out
                   else "(nog niet opgehaald)")
if st.sidebar.button("🔄 Ververs titelkansen"):
    if not api_key:
        st.sidebar.error("Geen API-key beschikbaar.")
    else:
        try:
            out = fetch_outright_odds(api_key)
            if out:
                st.session_state["outright_probs"] = out
                st.sidebar.success(f"{len(out)} landen opgehaald.")
            else:
                st.sidebar.warning("Geen titelkansen gevonden.")
        except Exception as e:
            st.sidebar.error(f"Fout bij ophalen: {e}")

weight_odds = st.sidebar.slider("Gewicht odds vs. model", 0.0, 1.0, 0.8, 0.05,
                                help="0.8 = 80% odds, 20% model. Geldt alleen voor "
                                     "wedstrijden waarvoor odds beschikbaar zijn.")

st.sidebar.subheader("Scoreregels poule")
exact = st.sidebar.number_input("Punten exacte uitslag", value=10, min_value=0)
goal_diff = st.sidebar.number_input("Punten doelsaldo", value=6, min_value=0)
winner = st.sidebar.number_input("Punten winnaar", value=4, min_value=0)
cumulative = st.sidebar.toggle("Cumulatief (interpretatie A)", value=False,
                               help="Aan = criteria stapelen (exact = 20). "
                                    "Uit = alleen de hoogste laag telt (interpretatie B).")
rules = ScoringRules(exact=exact, goal_diff=goal_diff, winner=winner, cumulative=cumulative)


# ----------------------------------------------------------------------------
# Data laden
# ----------------------------------------------------------------------------
ratings, cal, wc_records = load_everything()
wc = pd.DataFrame(wc_records)
odds_db = st.session_state["odds_db"]

st.title("⚽ WK 2026 Poule-voorspeller")
_blend_txt = (f"Odds-blend actief ({int(weight_odds*100)}% odds / "
              f"{int((1-weight_odds)*100)}% model) op {len(odds_db)} wedstrijden."
              if odds_db else "Geen odds geladen — puur model.")
st.caption(_blend_txt + f"  ·  Scoreregels: {winner}/{goal_diff}/{exact} "
           f"({'cumulatief' if cumulative else 'beste laag telt'}).")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Alle wedstrijden", "🔍 Wedstrijd-detail",
                                  "🏆 Toernooi-simulatie", "📝 Poule-advies"])


# ---- Tab 1: alle groepswedstrijden ----
with tab1:
    rows = []
    totaal_ev = 0.0
    for r in wc.itertuples():
        p = predict_match(ratings, r.home_team, r.away_team, cal, neutral=True,
                          city=r.city, odds_db=odds_db if odds_db else None,
                          weight_odds=weight_odds)
        m = p["matrix"]
        mi, mj = np.unravel_index(np.argmax(m), m.shape)
        oh, oa, ev = optimal_prediction(m, rules)
        totaal_ev += ev
        rows.append({
            "Wedstrijd": f"{r.home_team} – {r.away_team}",
            "Venue": r.city,
            "Modus": f"{mi}-{mj}",
            "Poule-advies": f"{oh}-{oa}",
            "EV": round(ev, 2),
            "1": round(p["p_home"], 2),
            "X": round(p["p_draw"], 2),
            "2": round(p["p_away"], 2),
            "Odds": "✓" if p["has_odds"] else "",
        })
    df_rows = pd.DataFrame(rows)
    c1, c2 = st.columns([3, 1])
    c1.metric("Verwachte totaalpunten groepsfase (poule-advies)", f"{totaal_ev:.1f}")
    c2.metric("Aantal wedstrijden", len(df_rows))
    st.dataframe(df_rows, use_container_width=True, hide_index=True, height=560)
    st.download_button("⬇️ Download als CSV", df_rows.to_csv(index=False).encode("utf-8"),
                       "wk2026_voorspellingen.csv", "text/csv")


# ---- Tab 2: wedstrijd-detail met heatmaps ----
with tab2:
    wc = wc.reset_index(drop=True)
    labels = [f"{r.home_team} – {r.away_team}  ({r.city})" for r in wc.itertuples()]
    idx = st.selectbox("Kies een wedstrijd", range(len(labels)),
                       format_func=lambda i: labels[i])
    row = wc.iloc[idx]
    p = predict_match(ratings, row.home_team, row.away_team, cal, neutral=True,
                      city=row.city, odds_db=odds_db if odds_db else None,
                      weight_odds=weight_odds)
    m = p["matrix"]
    mi, mj = np.unravel_index(np.argmax(m), m.shape)
    oh, oa, ev = optimal_prediction(m, rules)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"P({row.home_team})", f"{p['p_home']*100:.0f}%")
    k2.metric("P(gelijk)", f"{p['p_draw']*100:.0f}%")
    k3.metric(f"P({row.away_team})", f"{p['p_away']*100:.0f}%")
    k4.metric("Poule-advies", f"{oh}-{oa}", help=f"Modus is {mi}-{mj}")

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Kans per exacte uitslag**")
        st.plotly_chart(probability_heatmap(m, row.home_team, row.away_team),
                        use_container_width=True)
    with h2:
        st.markdown("**Verwachte punten per gok**")
        st.plotly_chart(expected_points_heatmap(m, rules, row.home_team, row.away_team),
                        use_container_width=True)


# ---- Tab 3: toernooi-simulatie ----
with tab3:
    st.write("Simuleert het hele toernooi met het **officiële R32-schema** "
             "(groepswinnaars vs nummers 3, runners-up onderling). De "
             "groepswedstrijden gebruiken de odds-blend uit de zijbalk; de "
             "knock-out blijft puur model (geen odds beschikbaar voor "
             "onbekende matchups).")
    n_sims = st.select_slider("Aantal simulaties", options=[5000, 10000, 20000, 50000],
                              value=20000)
    if st.button("▶️ Simuleer toernooi"):
        run_simulation(ratings, cal, wc_records, n_sims, odds_db, weight_odds)

    if "sim_result" in st.session_state:
        meta = st.session_state["sim_meta"]
        st.caption(f"{meta['n_sims']:,} simulaties · "
                   + (f"odds-blend {meta['weight']*100:.0f}% over {meta['n_odds']} wedstrijden"
                      if meta["with_odds"] else "puur model (geen odds gebruikt)"))
        df_res = pd.DataFrame(st.session_state["sim_result"]["stage_probs"])
        rename = {"P_winner": "Titel", "P_final": "Finale", "P_semi": "Halve",
                  "P_quarter": "Kwart", "P_last16": "R16", "P_last32": "R32"}
        for col, label in rename.items():
            df_res[label] = (df_res[col] * 100).round(1)

        # voeg bookmaker-titelkansen toe indien beschikbaar
        out_probs = st.session_state.get("outright_probs", {})
        has_out = bool(out_probs)
        if has_out:
            df_res["Bookmakers"] = df_res["team"].map(
                lambda t: round(out_probs.get(t, 0) * 100, 1))

        top = df_res.head(15)
        if has_out:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Simulatie", x=top["Titel"], y=top["team"],
                                 orientation="h", marker_color="#185FA5",
                                 text=top["Titel"].map(lambda v: f"{v}%")))
            fig.add_trace(go.Bar(name="Bookmakers", x=top["Bookmakers"], y=top["team"],
                                 orientation="h", marker_color="#E8A020",
                                 text=top["Bookmakers"].map(lambda v: f"{v}%")))
            fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"),
                              height=520, xaxis_title="Kans op de wereldtitel (%)",
                              margin=dict(l=10, r=10, t=20, b=10),
                              legend=dict(orientation="h", y=1.05))
        else:
            fig = go.Figure(go.Bar(
                x=top["Titel"], y=top["team"], orientation="h",
                text=top["Titel"].map(lambda v: f"{v}%"), marker_color="#185FA5"))
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500,
                              xaxis_title="Kans op de wereldtitel (%)",
                              margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        cols_show = ["team", "Titel"]
        if has_out:
            df_res["Verschil"] = (df_res["Titel"] - df_res["Bookmakers"]).round(1)
            cols_show += ["Bookmakers", "Verschil"]
        cols_show += ["Finale", "Halve", "Kwart", "R16", "R32"]
        if not has_out:
            st.caption("Haal de bookmaker-titelkansen op via '🔄 Ververs titelkansen' "
                       "in de zijbalk voor een directe vergelijking.")
        st.dataframe(df_res[cols_show], use_container_width=True,
                     hide_index=True, height=560)


# ---- Tab 4: volledig poule-advies ----
with tab4:
    st.write("Optimaal advies voor de extra rubrieken, op basis van de "
             "Monte-Carlo-simulatie (officieel R32-schema). De groepswedstrijden "
             "gebruiken de odds-blend uit de zijbalk; knock-out blijft puur model.")
    n_sims4 = st.select_slider("Aantal simulaties ", options=[5000, 10000, 20000, 50000],
                               value=20000, key="nsims_advies")
    if st.button("▶️ Genereer poule-advies"):
        run_simulation(ratings, cal, wc_records, n_sims4, odds_db, weight_odds)
        st.session_state["advice_generated"] = True

    if st.session_state.get("advice_generated") and "sim_result" in st.session_state:
        sim = st.session_state["sim_result"]
        meta = st.session_state["sim_meta"]
        st.caption(f"{meta['n_sims']:,} simulaties · "
                   + (f"odds-blend {meta['weight']*100:.0f}% over {meta['n_odds']} wedstrijden"
                      if meta["with_odds"] else "puur model (geen odds gebruikt)"))

        # 1. Groepstanden
        st.subheader("1 · Groepstanden  (2 pt exacte plek, 1 pt bij 1 plek afwijking)")
        standings_rows = []
        totaal_standen = 0.0
        for label in sorted(sim["position_probs"]):
            order, ev = recommend_group_standing(sim["position_probs"][label])
            totaal_standen += ev
            for i, team in enumerate(order, 1):
                p = sim["position_probs"][label][team]
                standings_rows.append({"Groep": label, "Plek": i, "Team": team,
                                       "P(deze plek)": f"{p[i-1]*100:.0f}%"})
        st.caption(f"Verwacht totaal: {totaal_standen:.1f} pt")
        st.dataframe(pd.DataFrame(standings_rows), use_container_width=True,
                     hide_index=True, height=400)

        # 2. Finalisten per fase
        st.subheader("2 · Finalisten per fase")
        quals, totaal_quals = recommend_qualifiers(sim["stage_probs"])
        labels = {"last32": "Laatste 32", "last16": "Achtste finale", "quarter": "Kwartfinale",
                  "semi": "Halve finale", "final": "Finale", "winner": "Winnaar"}
        out_probs = st.session_state.get("outright_probs", {})
        st.caption(f"Verwacht totaal: {totaal_quals:.1f} pt")
        for stage in ["last32", "last16", "quarter", "semi", "final", "winner"]:
            info = quals[stage]
            with st.expander(f"{labels[stage]} — kies {STAGE_COUNTS[stage]}  "
                             f"({STAGE_POINTS_DEFAULT[stage]} pt p.s., verwacht {info['ev']:.1f} pt)"):
                st.write(", ".join(info["picks"]))
                # voor de winnaar: extra vergelijkingstabel met bookmakers
                if stage == "winner" and out_probs:
                    st.markdown("---")
                    st.markdown("**Simulatie vs. bookmakers (top 10 WK-kansen)**")
                    sp = sorted(sim["stage_probs"], key=lambda x: -x["P_winner"])[:10]
                    cmp_rows = []
                    for r in sp:
                        t = r["team"]
                        sim_p = round(r["P_winner"] * 100, 1)
                        bk_p = round(out_probs.get(t, 0) * 100, 1)
                        diff = round(sim_p - bk_p, 1)
                        arrow = "↑" if diff > 1 else ("↓" if diff < -1 else "≈")
                        cmp_rows.append({"Team": t, "Simulatie %": sim_p,
                                         "Bookmakers %": bk_p, "Verschil": diff,
                                         "": arrow})
                    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True,
                                 use_container_width=True)

        # 3. Goaltotalen
        st.subheader("3 · Goaltotalen groepsfase (totaal over 3 wedstrijden)")
        g = sim["goals"]
        gf_rows = [{"Team": t, "Verw. goals": round(g[t]["gf"], 1),
                    "Kans koploper": f"{g[t]['p_most_gf']*100:.0f}%"}
                   for t in sorted(g, key=lambda x: -g[x]["gf"])[:8]]
        ga_rows = [{"Team": t, "Verw. tegengoals": round(g[t]["ga"], 1),
                    "Kans koploper": f"{g[t]['p_most_ga']*100:.0f}%"}
                   for t in sorted(g, key=lambda x: -g[x]["ga"])[:8]]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Meeste goals VOOR**")
            st.dataframe(pd.DataFrame(gf_rows), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Meeste goals TEGEN**")
            st.dataframe(pd.DataFrame(ga_rows), use_container_width=True, hide_index=True)

        # 4. Kaarten — session_state + ophaal-knop (werkt ook op de cloud)
        st.subheader("4 · Kaarten groepsfase")
        if "card_rates" not in st.session_state:
            st.session_state["card_rates"] = load_card_rates()   # leeg op cloud bij eerste run

        card_rates = st.session_state["card_rates"]
        af_key = get_secret("APIFOOTBALL_KEY")

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            if af_key:
                if st.button("🔄 Haal kaartdata op"):
                    with st.spinner("Team-ID's en kaart-tarieven ophalen via API-Football "
                                    "(~30 sec, ~96 API-calls) ..."):
                        teams_all = sorted(set(wc["home_team"]) | set(wc["away_team"]))
                        try:
                            ids = resolve_team_ids(af_key, teams_all)
                            rates = fetch_card_rates(af_key, ids)
                            st.session_state["card_rates"] = rates
                            st.success(f"{sum(1 for v in rates.values() if v['matches']>0)} "
                                       f"landen met echte data, rest op gemiddelde.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fout bij ophalen: {e}")
            else:
                st.caption("Geen API-Football-key beschikbaar.")
        with cc2:
            if not af_key:
                st.info("Voeg APIFOOTBALL_KEY toe in Secrets (cloud) of "
                        "`apifootball_key.txt` (lokaal) om kaartdata op te halen.")

        if card_rates:
            teams_all = sorted(set(wc["home_team"]) | set(wc["away_team"]))
            csim = simulate_cards(card_rates, teams_all)
            most_rows = [{"Team": t, "Kans koploper": f"{d['p_most']*100:.0f}%",
                          "Verw. kaartpunten": round(d["exp_card_points"], 1)}
                         for t, d in recommend_most_cards(csim, top=8)]
            few_rows = [{"Team": t, "Verw. punten": round(d["exp_fewest_points"], 2),
                         "Verw. geel": round(d["exp_yellow"], 1), "Verw. rood": round(d["exp_red"], 2)}
                        for t, d in recommend_fewest_cards(csim, top=8)]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Meeste kaarten** (1 pt geel, 2 pt rood)")
                st.dataframe(pd.DataFrame(most_rows), use_container_width=True, hide_index=True)
            with d2:
                st.markdown("**Minste kaarten** (5/3/1 pt)")
                st.dataframe(pd.DataFrame(few_rows), use_container_width=True, hide_index=True)
            st.caption("Kaarten zijn ruisig en scheidsrechter-afhankelijk — behandel dit "
                       "als ruwe indicatie, niet als sterke voorspelling.")

        # 5. Knock-out bracket volgens FIFA-schema
        st.subheader("5 · Knock-out bracket (volgens FIFA-schema, ingevuld door simulatie)")
        if "bracket" not in sim:
            st.error("De bracket-data ontbreekt in het simulatieresultaat. "
                     "Meestal komt dit doordat `poule_extras.py`, `simulate.py` of "
                     "`knockout.py` nog de oude versie is. Controleer dat alle vier "
                     "bestanden (app.py + die drie) in je GitHub-repo geüpdatet zijn, "
                     "en herstart de app via 'Manage app' → ⋮ → Reboot.")
        else:
            st.caption("Per slot het meest voorkomende team uit de simulatie. Het percentage "
                       "geeft aan hoe vaak dat team daar terechtkomt — lage % = onzekere positie.")
            components.html(render_bracket_html(sim["bracket"]), height=1180, scrolling=True)
