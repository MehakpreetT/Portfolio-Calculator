"""
Forward-Looking Portfolio Calculator — Streamlit Web App
----------------------------------------------------------
Run locally with:  streamlit run streamlit_app.py
Deploy for free on Streamlit Community Cloud by pushing this file
(+ requirements.txt) to a GitHub repo and connecting it there.

Tabs:
  1. Calculator      -> amount, risk profile, horizon -> target portfolio
  2. Strategy Guide   -> breakdown of the investment strategy behind
                         each risk profile

The market condition score is cached once per calendar day, so it only
recalculates the first time the app is opened after a new trading day's
data becomes available (i.e. it "updates" once per trading day rather
than on every click).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum
from datetime import date


# =======================================================================
# 1. CONFIG / CONSTANTS
# =======================================================================
class RiskProfile(Enum):
    CONSERVATIVE = "Conservative"
    NEUTRAL = "Neutral"
    GROWTH = "Growth"


STRATEGIC_MIX = {
    RiskProfile.CONSERVATIVE: {"cash": 0.02, "bonds": 0.58, "cdn_eq": 0.13, "us_eq": 0.15, "intl_eq": 0.12, "em_eq": 0.00},
    RiskProfile.NEUTRAL:      {"cash": 0.02, "bonds": 0.38, "cdn_eq": 0.15, "us_eq": 0.25, "intl_eq": 0.15, "em_eq": 0.05},
    RiskProfile.GROWTH:       {"cash": 0.02, "bonds": 0.23, "cdn_eq": 0.18, "us_eq": 0.30, "intl_eq": 0.19, "em_eq": 0.08},
}

TICKERS = {
    "cash": None,
    "bonds": "XBB.TO",
    "cdn_eq": "XIC.TO",
    "us_eq": "VFV.TO",
    "intl_eq": "XEF.TO",
    "em_eq": "XEC.TO",
}

ASSET_LABELS = {
    "cash": "Cash",
    "bonds": "Fixed Income",
    "cdn_eq": "Canadian Equity",
    "us_eq": "U.S. Equity",
    "intl_eq": "International Equity",
    "em_eq": "Emerging Markets",
}

TACTICAL_RANGE = 0.15

STRATEGY_NOTES = {
    RiskProfile.CONSERVATIVE: {
        "summary": "Prioritizes capital preservation with modest growth. Best suited for short-to-medium horizons or investors uncomfortable with large swings in value.",
        "philosophy": "The majority of the portfolio sits in fixed income to dampen volatility, with a small equity sleeve to keep pace with inflation. Emerging markets are excluded entirely to avoid the sharpest drawdowns.",
        "expected_volatility": "Low",
        "who_its_for": "Investors nearing a financial goal, or anyone who would be tempted to sell during a downturn.",
    },
    RiskProfile.NEUTRAL: {
        "summary": "Balances growth and stability. The default choice for medium-to-long horizons without a strong view on risk tolerance.",
        "philosophy": "Roughly 60/40 equities-to-fixed-income split, diversified across Canadian, U.S., international, and emerging markets. Aims to participate meaningfully in equity growth while still cushioning downturns with bonds.",
        "expected_volatility": "Moderate",
        "who_its_for": "Investors with a multi-year horizon who want growth but aren't chasing maximum returns.",
    },
    RiskProfile.GROWTH: {
        "summary": "Prioritizes long-term capital growth over stability. Best suited for long horizons where short-term volatility can be absorbed.",
        "philosophy": "Heavily weighted toward equities, including a larger emerging markets sleeve for higher (but more volatile) growth potential. Fixed income is minimal, used mainly to reduce extreme swings rather than provide income.",
        "expected_volatility": "High",
        "who_its_for": "Younger investors or anyone with a long time horizon and high tolerance for drawdowns.",
    },
}


# =======================================================================
# 2. LOGIC (same model as before)
# =======================================================================
def horizon_tilt(years: float) -> float:
    if years >= 15:
        return 1.0
    elif years >= 7:
        return 0.5
    elif years >= 3:
        return 0.0
    else:
        return -0.75


@st.cache_data(ttl=60 * 60 * 12)  # refresh at most twice a day; keyed by date below for a clean "once per trading day" feel
def market_condition_score(cache_key: str):
    """
    cache_key is today's date string — Streamlit will only re-run this
    function once per unique cache_key, effectively refreshing once per day
    (and picking up the newly completed previous trading day's data).
    """
    try:
        vix_hist = yf.Ticker("^VIX").history(period="10d")
        spx_hist = yf.Ticker("^GSPC").history(period="10d")

        vix_hist = vix_hist.iloc[:-1] if len(vix_hist) > 1 else vix_hist
        spx_hist = spx_hist.iloc[:-1] if len(spx_hist) > 1 else spx_hist

        prev_day_vix_open = vix_hist["Open"].iloc[-1]
        prev_day_spx_open = spx_hist["Open"].iloc[-1]
        two_days_ago_spx_open = spx_hist["Open"].iloc[-2]
        as_of_date = vix_hist.index[-1].strftime("%Y-%m-%d")

        momentum = (prev_day_spx_open / two_days_ago_spx_open) - 1
        vix_score = max(-1.0, min(1.0, (20 - prev_day_vix_open) / 10))
        momentum_score = max(-1.0, min(1.0, momentum / 0.05))

        score = round(0.5 * vix_score + 0.5 * momentum_score, 2)
        return score, as_of_date, prev_day_vix_open
    except Exception as e:
        return None, str(e), None


def build_portfolio(amount, risk_profile, horizon_years, market_score):
    base = STRATEGIC_MIX[risk_profile].copy()
    h_tilt = horizon_tilt(horizon_years)
    m_score = market_score if market_score is not None else 0.0

    composite_tilt = 0.65 * h_tilt + 0.35 * m_score
    composite_tilt = max(-1.0, min(1.0, composite_tilt))

    equity_keys = ["cdn_eq", "us_eq", "intl_eq", "em_eq"]
    defensive_keys = ["bonds", "cash"]

    shift = composite_tilt * TACTICAL_RANGE
    total_equity_base = sum(base[k] for k in equity_keys)
    total_defensive_base = sum(base[k] for k in defensive_keys)
    shift = max(min(shift, total_defensive_base * 0.9), -total_equity_base * 0.9)

    adjusted = base.copy()
    if total_equity_base > 0:
        for k in equity_keys:
            adjusted[k] = base[k] + shift * (base[k] / total_equity_base)
    if total_defensive_base > 0:
        for k in defensive_keys:
            adjusted[k] = base[k] - shift * (base[k] / total_defensive_base)

    total = sum(adjusted.values())
    weights = {k: v / total for k, v in adjusted.items()}
    dollar_allocation = {k: round(amount * v, 2) for k, v in weights.items()}
    return weights, dollar_allocation


# =======================================================================
# 3. PAGE CONFIG + STYLE
# =======================================================================
st.set_page_config(page_title="Portfolio Calculator", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        h1, h2, h3 { color: #e8eaed; }
        .metric-card {
            background-color: #1c1f26;
            border-radius: 12px;
            padding: 18px;
            border: 1px solid #2b2f38;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5em 1.5em;
        }
        .stButton>button:hover {
            background-color: #34495e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Forward-Looking Portfolio Calculator")
st.caption("Builds a target asset allocation from your risk profile, horizon, and the prior trading day's market conditions.")

tab1, tab2 = st.tabs(["🧮 Calculator", "📘 Strategy Guide"])


# =======================================================================
# 4. TAB 1 — CALCULATOR
# =======================================================================
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Investment Amount ($)", min_value=1.0, value=20000.0, step=500.0)
    with col2:
        risk_choice = st.selectbox("Risk Profile", [p.value for p in RiskProfile], index=1)
    with col3:
        horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=99, value=10, step=1)

    calculate = st.button("Calculate Portfolio", use_container_width=False)

    st.divider()

    score, as_of, vix_level = market_condition_score(str(date.today()))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if score is not None:
            st.metric("Market Condition Score", f"{score:+.2f}", help="Range: -1 (risk-off) to +1 (risk-on)")
        else:
            st.metric("Market Condition Score", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("As-of Date (9:00 AM open)", as_of if score is not None else "unavailable")
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prior-Day VIX (open)", f"{vix_level:.2f}" if vix_level else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    if calculate:
        risk_profile = RiskProfile(risk_choice)
        weights, dollar_allocation = build_portfolio(amount, risk_profile, horizon, score)

        st.subheader("Recommended Allocation")

        left, right = st.columns([1, 1.3])

        with left:
            fig = go.Figure(data=[go.Pie(
                labels=[ASSET_LABELS[k] for k in weights],
                values=[v * 100 for v in weights.values()],
                hole=0.45,
                textinfo="label+percent",
            )])
            fig.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            df = pd.DataFrame({
                "Asset Class": [ASSET_LABELS[k] for k in weights],
                "Ticker": [TICKERS[k] or "—" for k in weights],
                "Weight": [f"{v*100:.1f}%" for v in weights.values()],
                "Dollar Amount": [f"${dollar_allocation[k]:,.2f}" for k in weights],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            total = sum(dollar_allocation.values())
            st.markdown(f"**Total: ${total:,.2f}**")


# =======================================================================
# 5. TAB 2 — STRATEGY GUIDE
# =======================================================================
with tab2:
    st.subheader("Strategy Breakdown by Risk Profile")
    st.caption("What each profile is built for, and the reasoning behind the weights.")

    for profile in RiskProfile:
        notes = STRATEGY_NOTES[profile]
        mix = STRATEGIC_MIX[profile]

        with st.expander(f"**{profile.value}** — {notes['summary']}", expanded=(profile == RiskProfile.NEUTRAL)):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                fig = go.Figure(data=[go.Pie(
                    labels=[ASSET_LABELS[k] for k in mix],
                    values=[v * 100 for v in mix.values()],
                    hole=0.45,
                    textinfo="label+percent",
                )])
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaed"),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown(f"**Philosophy:** {notes['philosophy']}")
                st.markdown(f"**Expected Volatility:** {notes['expected_volatility']}")
                st.markdown(f"**Best suited for:** {notes['who_its_for']}")

    st.divider()
    st.markdown("""
        **How the tactical tilt works:** Each profile's weights above are the *strategic* baseline.
        On the Calculator tab, two things nudge the final allocation within a ±15% range:
        - **Horizon** — longer horizons tilt toward equities, shorter horizons tilt toward safety
        - **Market conditions** — calculated from the previous trading day's VIX level and
          S&P 500 momentum at market open; calmer/rising markets tilt toward equities,
          elevated volatility or falling markets tilt toward bonds and cash
    """)
