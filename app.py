"""
app.py
======
IPL Match Winner Predictor -- a Streamlit front end for the Random Forest
pipeline trained in `train_model.py` (reproducing the modeling logic
originally developed in `notebook.ipynb`).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & page config (set_page_config must be the first Streamlit call)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "pipeline.pkl"

st.set_page_config(
    page_title="IPL Match Winner Predictor",
    page_icon="\U0001F3CF",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens -- kept in one place so the theme is easy to retune
# ---------------------------------------------------------------------------
COLOR_BG = "#0B1220"
COLOR_CARD = "#141D33"
COLOR_CARD_BORDER = "rgba(255, 255, 255, 0.08)"
COLOR_GOLD = "#F2B705"
COLOR_TEXT = "#E9ECF4"
COLOR_MUTED = "#8D97AF"
COLOR_TRACK = "#28304A"

# Approximate franchise brand colors, used only as flat-color badge accents
# (not logos/crests) so no third-party artwork or trademarked imagery is
# reproduced anywhere in the app.
TEAM_COLORS = {
    "Mumbai Indians": "#2E7BDB",
    "Chennai Super Kings": "#FFC72C",
    "Royal Challengers Bangalore": "#E7343A",
    "Kolkata Knight Riders": "#8B5CF6",
    "Delhi Daredevils": "#3B82F6",
    "Kings XI Punjab": "#C0392B",
    "Rajasthan Royals": "#EC4899",
    "Deccan Chargers": "#2CA6A4",
    "Sunrisers Hyderabad": "#F97316",
    "Pune Warriors": "#94A3B8",
    "Gujarat Lions": "#E4572E",
    "Kochi Tuskers Kerala": "#A855F7",
    "Rising Pune Supergiants": "#38BDF8",
}
DEFAULT_TEAM_COLOR = "#6B7A99"

TEAM_INITIALS = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bangalore": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Daredevils": "DD",
    "Kings XI Punjab": "KXIP",
    "Rajasthan Royals": "RR",
    "Deccan Chargers": "DC",
    "Sunrisers Hyderabad": "SRH",
    "Pune Warriors": "PW",
    "Gujarat Lions": "GL",
    "Kochi Tuskers Kerala": "KTK",
    "Rising Pune Supergiants": "RPS",
}

FEATURE_ORDER = ["season", "city", "team1", "team2", "toss_winner", "toss_decision"]
FEATURE_LABELS = {
    "season": "\U0001F4C5 Season",
    "city": "\U0001F4CD City",
    "team1": "\U0001F3CF Team 1",
    "team2": "\U0001F3CF Team 2",
    "toss_winner": "\U0001FA99 Toss Winner",
    "toss_decision": "\U0001F3AF Toss Decision",
}

# Mirrors the example scenario from the original notebook, so the form opens
# on a recognizable, previously-seen match-up rather than an arbitrary one.
DEFAULT_SCENARIO = {
    "season": 2010,
    "city": "Chennai",
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "toss_winner": "Chennai Super Kings",
    "toss_decision": "bat",
}


# ---------------------------------------------------------------------------
# Small pure-Python helpers
# ---------------------------------------------------------------------------
def team_color(name: str) -> str:
    return TEAM_COLORS.get(name, DEFAULT_TEAM_COLOR)


def team_initials(name: str) -> str:
    if name in TEAM_INITIALS:
        return TEAM_INITIALS[name]
    words = [w for w in name.split() if w.lower() not in {"xi", "of"}]
    return "".join(w[0] for w in words[:4]).upper()


def readable_text_color(hex_color: str) -> str:
    """Pick near-black or near-white text so it stays readable on `hex_color`."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#101418" if luminance > 0.6 else "#F8F9FC"


def option_index(options: list, value, fallback: int = 0) -> int:
    """Index of `value` in `options`, or `fallback` if it isn't present."""
    try:
        return options.index(value)
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# HTML snippets (small, self-contained -- each call renders a complete block)
# ---------------------------------------------------------------------------
def render_badge_html(name: str, size: int = 44, font_size: float = 0.95) -> str:
    color = team_color(name)
    text_color = readable_text_color(color)
    initials = team_initials(name)
    return (
        f'<div class="team-badge" style="width:{size}px;height:{size}px;'
        f'background:{color};color:{text_color};font-size:{font_size}rem;">'
        f"{initials}</div>"
    )


def render_fixture_html(team1: str, team2: str) -> str:
    return (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'gap:0.9rem;margin:0.5rem 0 0.1rem 0;">'
        f'{render_badge_html(team1, size=40, font_size=0.85)}'
        f'<span style="color:{COLOR_MUTED};font-family:\'Teko\',sans-serif;'
        f'font-size:1.15rem;letter-spacing:.08em;">VS</span>'
        f'{render_badge_html(team2, size=40, font_size=0.85)}'
        "</div>"
    )


def block_bar_html(pct: float, color: str, total_blocks: int = 26) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = max(0, min(total_blocks, round(pct / 100 * total_blocks)))
    empty = total_blocks - filled
    return (
        '<span class="score-bar">'
        f'<span style="color:{color};">{"\u2588" * filled}</span>'
        f'<span style="color:{COLOR_TRACK};">{"\u2591" * empty}</span>'
        "</span>"
    )


def render_score_row_html(name: str, pct: float) -> str:
    color = team_color(name)
    return (
        '<div class="score-row">'
        '<div class="score-label">'
        f"<span>{name}</span>"
        f'<span class="score-pct" style="color:{color};">{pct:.1f}%</span>'
        "</div>"
        f"{block_bar_html(pct, color)}"
        "</div>"
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Teko:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 0%, rgba(242, 183, 5, 0.08), transparent 30%),
                radial-gradient(circle at 88% 0%, rgba(46, 123, 219, 0.10), transparent 35%),
                {COLOR_BG};
        }}

        .hero-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: {COLOR_GOLD};
            font-size: 0.72rem;
            font-weight: 500;
        }}
        .hero-title {{
            font-family: 'Teko', sans-serif;
            font-weight: 700;
            font-size: 3.4rem;
            line-height: 1.05;
            color: {COLOR_TEXT};
            margin: 0.15rem 0 0.25rem 0;
        }}
        .hero-subtitle {{ color: {COLOR_MUTED}; font-size: 1.03rem; margin-bottom: 0.2rem; }}

        .section-heading {{
            font-family: 'Teko', sans-serif;
            font-size: 1.7rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: {COLOR_TEXT};
            margin: 1.6rem 0 0.5rem 0;
        }}

        .card-heading {{
            font-family: 'Teko', sans-serif;
            font-size: 1.35rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            color: {COLOR_TEXT};
            margin: 0 0 0.8rem 0;
        }}

        .team-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-family: 'Teko', sans-serif;
            font-weight: 600;
            flex-shrink: 0;
        }}

        .score-row {{ margin-bottom: 0.85rem; }}
        .score-row:last-child {{ margin-bottom: 0; }}
        .score-label {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            font-size: 0.95rem;
            color: {COLOR_TEXT};
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}
        .score-pct {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
        .score-bar {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            letter-spacing: -1px;
            line-height: 1;
            word-break: break-all;
        }}

        .ipl-footer {{
            text-align: center;
            color: {COLOR_MUTED};
            font-size: 0.85rem;
            padding: 1.6rem 0 0.6rem 0;
            border-top: 1px solid {COLOR_CARD_BORDER};
            margin-top: 2rem;
        }}
        .ipl-footer a {{ color: {COLOR_GOLD}; text-decoration: none; }}

        [class*="st-key-"] {{
            background: {COLOR_CARD} !important;
            border-color: {COLOR_CARD_BORDER} !important;
            border-radius: 16px !important;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.02);
            border: 1px solid {COLOR_CARD_BORDER};
            border-radius: 12px;
            padding: 0.6rem 0.8rem 0.4rem 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached data / model loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_match_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_dropdown_options(df: pd.DataFrame) -> tuple[list, list, list, list]:
    """Populate every dropdown straight from the dataset -- no hardcoded lists."""
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    cities = sorted(df["city"].dropna().unique().tolist())
    teams = sorted(set(df["team1"].dropna().unique()) | set(df["team2"].dropna().unique()))
    toss_decisions = sorted(df["toss_decision"].dropna().unique().tolist())
    return seasons, cities, teams, toss_decisions


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def build_input_frame(season, city, team1, team2, toss_winner, toss_decision) -> pd.DataFrame:
    """Builds a single-row dataframe with the exact column layout the
    pipeline was trained on (see train_model.py / notebook.ipynb)."""
    return pd.DataFrame(
        [[season, city, team1, team2, toss_winner, toss_decision]],
        columns=FEATURE_ORDER,
    )


def predict_match(pipeline, input_df: pd.DataFrame):
    """Runs the pipeline once and returns its raw probability vector.

    Deliberately does NOT return `pipeline.predict(...)` as "the winner".
    The pipeline is a 13-class classifier (it can output any of the 13
    franchises, not just the two selected here), and testing this app
    surfaced a scenario where the unconstrained argmax named a team that
    wasn't even playing in the fixture -- e.g. for an unusual team/city
    combination, the model can favor "whoever usually wins in this city"
    over the two teams actually selected. See head_to_head_probabilities(),
    which is what actually decides the winner shown to the user.
    """
    proba = pipeline.predict_proba(input_df)[0]
    classes = list(pipeline.classes_)
    return proba, classes


def head_to_head_probabilities(proba, classes: list, team1: str, team2: str):
    """Extract and renormalize predict_proba() for just the two competing teams.

    The pipeline is a 13-class classifier, so raw predict_proba() spreads a
    little probability mass across teams that aren't even playing in this
    fixture. The original notebook tried to isolate the top two values with
    `prob.sort()`, but a plain array sort reorders the probabilities without
    reordering the team labels, so the printed numbers couldn't be reliably
    matched back to a team. Here we look each team up by name directly in
    `pipeline.classes_`, then renormalize across just those two so the two
    percentages shown to the user always sum to 100%.
    """
    p1_raw = proba[classes.index(team1)] if team1 in classes else 0.0
    p2_raw = proba[classes.index(team2)] if team2 in classes else 0.0
    total = p1_raw + p2_raw
    if total <= 0:
        return 50.0, 50.0, p1_raw, p2_raw
    return (p1_raw / total) * 100, (p2_raw / total) * 100, p1_raw, p2_raw


def pick_winner(team1: str, team2: str, p1_raw: float, p2_raw: float) -> str:
    """The predicted winner is always one of the two selected teams: whichever
    has the higher head-to-head probability. This is intentionally NOT the
    same as `pipeline.predict(...)` -- see predict_match() docstring."""
    return team1 if p1_raw >= p2_raw else team2


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_sidebar(bundle: dict) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="hero-eyebrow">Match Center</div>'
            '<div class="hero-title" style="font-size:2.1rem;">IPL Predictor</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True, key="about-card"):
            st.markdown('<div class="card-heading">About</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:{COLOR_MUTED}; font-size:0.9rem; line-height:1.55;">'
                "Predicts the winner of an IPL match from pre-match conditions only: "
                "season, host city, the two teams, and the toss. Trained on historical "
                "IPL results with a Random Forest classifier."
                "</p>",
                unsafe_allow_html=True,
            )

        with st.container(border=True, key="accuracy-card"):
            st.markdown('<div class="card-heading">Model Accuracy</div>', unsafe_allow_html=True)
            st.metric("Test-set accuracy", f"{bundle['accuracy'] * 100:.1f}%")
            st.caption(
                f"{bundle['model_name']} \u00b7 {bundle['n_train']} train / "
                f"{bundle['n_test']} test matches"
            )

        with st.container(border=True, key="author-card"):
            st.markdown('<div class="card-heading">Author</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:{COLOR_TEXT}; font-size:0.92rem; margin-bottom:0.15rem;">Aman Kumar</p>'
                f'<p style="color:{COLOR_MUTED}; font-size:0.85rem;">ML Enthusiast</p>',
                unsafe_allow_html=True,
            )
            st.link_button(
                "\U0001F517 View on GitHub",
                "https://github.com/Amank0106/IPL_Match_prediction_model",
                width="stretch",
            )
            


def render_hero() -> None:
    st.markdown(
        '<div class="hero-eyebrow">Machine Learning \u00b7 Indian Premier League</div>'
        '<div class="hero-title">\U0001F3CF IPL Match Winner Predictor</div>'
        '<div class="hero-subtitle">Predict the winner of an IPL match using Machine Learning.</div>',
        unsafe_allow_html=True,
    )


def render_input_form(seasons, cities, teams, toss_decisions) -> tuple:
    with st.container(border=True, key="input-card"):
        st.markdown('<div class="card-heading">\U0001F3DF\uFE0F Match Setup</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            season = st.selectbox(
                "\U0001F4C5 Season",
                seasons,
                index=option_index(seasons, DEFAULT_SCENARIO["season"]),
                key="season_sel",
            )
        with c2:
            city = st.selectbox(
                "\U0001F4CD City",
                cities,
                index=option_index(cities, DEFAULT_SCENARIO["city"]),
                key="city_sel",
            )

        t1c, t2c = st.columns(2)
        with t1c:
            team1 = st.selectbox(
                "\U0001F3CF Team 1",
                teams,
                index=option_index(teams, DEFAULT_SCENARIO["team1"]),
                key="team1_sel",
            )
        with t2c:
            team2 = st.selectbox(
                "\U0001F3CF Team 2",
                teams,
                index=option_index(teams, DEFAULT_SCENARIO["team2"]),
                key="team2_sel",
            )

        if team1 == team2:
            st.warning("Team 1 and Team 2 are the same \u2014 pick two different teams.")
        else:
            st.markdown(render_fixture_html(team1, team2), unsafe_allow_html=True)

        twc, tdc = st.columns(2)
        with twc:
            toss_winner = st.selectbox(
                "\U0001FA99 Toss Winner",
                teams,
                index=option_index(teams, DEFAULT_SCENARIO["toss_winner"]),
                key="toss_winner_sel",
            )
            if team1 != team2 and toss_winner not in (team1, team2):
                st.caption(f"\u26A0\uFE0F Should be {team1} or {team2}")
        with tdc:
            toss_decision = st.selectbox(
                "\U0001F3AF Toss Decision",
                toss_decisions,
                format_func=lambda v: str(v).capitalize(),
                index=option_index(toss_decisions, DEFAULT_SCENARIO["toss_decision"]),
                key="toss_decision_sel",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col, reset_col = st.columns([3, 1])
        with btn_col:
            predict_clicked = st.button(
                "\U0001F52E Predict Winner", type="primary", width="stretch"
            )
        with reset_col:
            reset_clicked = st.button("\U0001F501 Reset", width="stretch")

    return season, city, team1, team2, toss_winner, toss_decision, predict_clicked, reset_clicked


def render_probability_chart(team1: str, p1: float, team2: str, p2: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.3))
    fig.patch.set_facecolor(COLOR_CARD)
    ax.set_facecolor(COLOR_CARD)

    labels = [team2, team1]
    values = [p2, p1]
    colors = [team_color(team2), team_color(team1)]

    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Win Probability (%)", color=COLOR_MUTED, fontsize=9)
    ax.tick_params(colors=COLOR_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color=COLOR_TRACK, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.1f%%", color=COLOR_TEXT, padding=4, fontsize=10, fontweight="bold")
    fig.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_result(result: dict) -> None:
    team1, team2, winner = result["team1"], result["team2"], result["winner"]
    p1, p2 = result["p1"], result["p2"]

    st.markdown('<div class="section-heading">\U0001F3C6 Prediction</div>', unsafe_allow_html=True)

    badge_col, msg_col = st.columns([1, 8], vertical_alignment="center")
    with badge_col:
        st.markdown(render_badge_html(winner, size=50, font_size=1.0), unsafe_allow_html=True)
    with msg_col:
        st.success(f"**Predicted Winner** \u2014 {winner}", icon="\U0001F3C6")

    with st.container(border=True, key="probability-card"):
        st.markdown('<div class="card-heading">Winning Probability</div>', unsafe_allow_html=True)
        rows_html = render_score_row_html(team1, p1) + render_score_row_html(team2, p2)
        st.markdown(rows_html, unsafe_allow_html=True)
        st.caption(
            "Probabilities are renormalized across Team 1 and Team 2 only, "
            "so they sum to 100% for this match-up."
        )

        with st.expander("\U0001F4CA View probability chart"):
            render_probability_chart(team1, p1, team2, p2)


def render_model_information(bundle: dict) -> None:
    st.markdown('<div class="section-heading">\U0001F4CA Model Information</div>', unsafe_allow_html=True)
    with st.container(border=True, key="model-info-card"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Algorithm Used", bundle["model_name"])
        m2.metric("Accuracy", f"{bundle['accuracy'] * 100:.1f}%")
        m3.metric("Matches Used", bundle["n_matches_used"])

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        st.markdown(f'<p style="color:{COLOR_TEXT}; font-weight:600; margin-bottom:0.5rem;">Input Features</p>', unsafe_allow_html=True)
        feat_cols = st.columns(len(bundle["feature_order"]))
        for col, feat in zip(feat_cols, bundle["feature_order"]):
            col.markdown(
                f'<div style="text-align:center; background:rgba(255,255,255,0.03); '
                f'border:1px solid {COLOR_CARD_BORDER}; border-radius:10px; padding:0.5rem 0.3rem; '
                f'color:{COLOR_TEXT}; font-size:0.82rem;">{FEATURE_LABELS.get(feat, feat)}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("How was accuracy measured?"):
            st.write(
                f"The pipeline was evaluated on a held-out test split of "
                f"{bundle['n_test']} matches (20% of {bundle['n_matches_used']} usable "
                f"matches, with the remaining {bundle['n_train']} used for training). "
                "Because the model only sees information known *before* the match -- "
                "season, city, the two teams, and the toss -- its accuracy reflects "
                "how predictable IPL outcomes are from fixture data alone, without "
                "any in-match signals such as scores or wickets."
            )


def render_history() -> None:
    history = st.session_state.get("history", [])
    with st.expander(f"\U0001F553 Prediction History ({len(history)})", expanded=False):
        if not history:
            st.caption("No predictions yet this session \u2014 make a prediction to see it appear here.")
            return
        hist_df = pd.DataFrame(list(reversed(history)))
        st.dataframe(hist_df, width="stretch", hide_index=True)
        if st.button("\U0001F5D1\uFE0F Clear history", key="clear_history_btn"):
            st.session_state["history"] = []
            st.rerun()


def render_footer() -> None:
    st.markdown(
        '<div class="ipl-footer">Made with \u2764\uFE0F using Streamlit and Scikit-Learn '
        '&middot; <a href="https://github.com/your-username/ipl-match-analysis">GitHub</a></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Form reset
# ---------------------------------------------------------------------------
def reset_form() -> None:
    for key in (
        "season_sel",
        "city_sel",
        "team1_sel",
        "team2_sel",
        "toss_winner_sel",
        "toss_decision_sel",
        "last_result",
    ):
        st.session_state.pop(key, None)
    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()

    match_df = load_match_data()
    seasons, cities, teams, toss_decisions = get_dropdown_options(match_df)
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]

    render_sidebar(bundle)
    render_hero()

    (
        season,
        city,
        team1,
        team2,
        toss_winner,
        toss_decision,
        predict_clicked,
        reset_clicked,
    ) = render_input_form(seasons, cities, teams, toss_decisions)

    if reset_clicked:
        reset_form()

    if predict_clicked:
        if team1 == team2:
            st.warning("\u26A0\uFE0F Team 1 and Team 2 must be different. Please choose two distinct teams.")
        elif toss_winner not in (team1, team2):
            st.warning(f"\u26A0\uFE0F Toss winner must be either **{team1}** or **{team2}**.")
        else:
            with st.spinner("\U0001F3CF Analyzing pitch, toss, and head-to-head history..."):
                time.sleep(0.7)  # brief pause so the spinner is perceptible
                input_df = build_input_frame(season, city, team1, team2, toss_winner, toss_decision)
                proba, classes = predict_match(pipeline, input_df)
                p1, p2, p1_raw, p2_raw = head_to_head_probabilities(proba, classes, team1, team2)
                winner = pick_winner(team1, team2, p1_raw, p2_raw)

            result = {
                "season": season,
                "city": city,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "p1": p1,
                "p2": p2,
            }
            st.session_state["last_result"] = result
            st.session_state.setdefault("history", []).append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Team 1": team1,
                    "Team 2": team2,
                    "Toss": f"{toss_winner} ({toss_decision})",
                    "Predicted Winner": winner,
                    "Confidence": f"{max(p1, p2):.1f}%",
                }
            )

    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])

    render_model_information(bundle)
    render_history()
    render_footer()


if __name__ == "__main__":
    main()
