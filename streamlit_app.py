"""
PortPicker — Forward-Looking Portfolio Construction Web App
--------------------------------------------------------------
Run locally with:  streamlit run streamlit_app.py

Tabs:
  1. Calculator          -> build a recommended portfolio, then optionally
                            override weights manually and save it
  2. Backtest & Risk      -> historical performance + Sharpe ratio for the
                            current portfolio vs. a 60/40 benchmark
  3. Compare Profiles     -> Conservative / Neutral / Growth side-by-side
  4. Market News          -> recent headlines per asset class (via yfinance)
  5. Saved Portfolios     -> view/manage portfolios you've saved
  6. Strategy Guide       -> philosophy behind each risk profile

New asset classes added vs. the original 6-sleeve version:
  - Gold / Commodities (CGL.TO)
  - REITs (XRE.TO)
These add diversification beyond pure equities/bonds and are standard
inclusions in most real multi-asset portfolios.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from enum import Enum
from datetime import date

SAVE_FILE = "saved_portfolios.json"
RISK_FREE_RATE = 0.03  # annualized, used for Sharpe ratio calc


# =======================================================================
# 1. CONFIG / CONSTANTS
# =======================================================================
class RiskProfile(Enum):
    CONSERVATIVE = "Conservative"
    NEUTRAL = "Neutral"
    GROWTH = "Growth"


# Updated to include Gold/Commodities and REITs as two new sleeves
STRATEGIC_MIX = {
    RiskProfile.CONSERVATIVE: {"cash": 0.02, "bonds": 0.53, "cdn_eq": 0.12, "us_eq": 0.13, "intl_eq": 0.10, "em_eq": 0.00, "gold": 0.05, "reit": 0.05},
    RiskProfile.NEUTRAL:      {"cash": 0.02, "bonds": 0.33, "cdn_eq": 0.13, "us_eq": 0.22, "intl_eq": 0.13, "em_eq": 0.04, "gold": 0.05, "reit": 0.08},
    RiskProfile.GROWTH:       {"cash": 0.02, "bonds": 0.18, "cdn_eq": 0.15, "us_eq": 0.26, "intl_eq": 0.16, "em_eq": 0.07, "gold": 0.06, "reit": 0.10},
}

TICKERS = {
    "cash": None,
    "bonds": "XBB.TO",
    "cdn_eq": "XIC.TO",
    "us_eq": "VFV.TO",
    "intl_eq": "XEF.TO",
    "em_eq": "XEC.TO",
    "gold": "CGL.TO",
    "reit": "XRE.TO",
}

ASSET_LABELS = {
    "cash": "Cash",
    "bonds": "Fixed Income",
    "cdn_eq": "Canadian Equity",
    "us_eq": "U.S. Equity",
    "intl_eq": "International Equity",
    "em_eq": "Emerging Markets",
    "gold": "Gold / Commodities",
    "reit": "REITs",
}

TACTICAL_RANGE = 0.15

STRATEGY_NOTES = {
    RiskProfile.CONSERVATIVE: {
        "summary": "Prioritizes capital preservation with modest growth.",
        "philosophy": "The majority of the portfolio sits in fixed income to dampen volatility, with small equity, gold, and REIT sleeves for diversification and inflation protection. Emerging markets are excluded to avoid the sharpest drawdowns.",
        "expected_volatility": "Low",
        "who_its_for": "Investors nearing a financial goal, or anyone who would be tempted to sell during a downturn.",
    },
    RiskProfile.NEUTRAL: {
        "summary": "Balances growth and stability.",
        "philosophy": "Roughly 60/40 growth-to-defensive split, diversified across regions plus gold and REIT sleeves to smooth out equity/bond correlation risk.",
        "expected_volatility": "Moderate",
        "who_its_for": "Investors with a multi-year horizon who want growth but aren't chasing maximum returns.",
    },
    RiskProfile.GROWTH: {
        "summary": "Prioritizes long-term capital growth over stability.",
        "philosophy": "Heavily weighted toward equities and real assets (REITs, gold) for higher long-run growth potential. Fixed income is minimal, used mainly to reduce extreme swings.",
        "expected_volatility": "High",
        "who_its_for": "Younger investors or anyone with a long time horizon and high tolerance for drawdowns.",
    },
}


# =======================================================================
# 2. CORE LOGIC
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


@st.cache_data(ttl=60 * 60 * 12)
def market_condition_score(cache_key: str):
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


def build_portfolio(risk_profile, horizon_years, market_score):
    base = STRATEGIC_MIX[risk_profile].copy()
    h_tilt = horizon_tilt(horizon_years)
    m_score = market_score if market_score is not None else 0.0

    composite_tilt = 0.65 * h_tilt + 0.35 * m_score
    composite_tilt = max(-1.0, min(1.0, composite_tilt))

    equity_like_keys = ["cdn_eq", "us_eq", "intl_eq", "em_eq", "reit"]
    defensive_keys = ["bonds", "cash", "gold"]

    shift = composite_tilt * TACTICAL_RANGE
    total_equity_base = sum(base[k] for k in equity_like_keys)
    total_defensive_base = sum(base[k] for k in defensive_keys)
    shift = max(min(shift, total_defensive_base * 0.9), -total_equity_base * 0.9)

    adjusted = base.copy()
    if total_equity_base > 0:
        for k in equity_like_keys:
            adjusted[k] = base[k] + shift * (base[k] / total_equity_base)
    if total_defensive_base > 0:
        for k in defensive_keys:
            adjusted[k] = base[k] - shift * (base[k] / total_defensive_base)

    total = sum(adjusted.values())
    weights = {k: v / total for k, v in adjusted.items()}
    return weights


# =======================================================================
# 3. BACKTEST + SHARPE RATIO
# =======================================================================
@st.cache_data(ttl=60 * 60 * 6)
def fetch_price_history(tickers_tuple, period):
    tickers = [t for t in tickers_tuple if t is not None]
    data = yf.download(tickers, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


def run_backtest(weights: dict, years: int):
    period = "1y" if years <= 1 else ("5y" if years <= 5 else "10y")
    tickers_used = {k: TICKERS[k] for k in weights if TICKERS[k] is not None}

    try:
        prices = fetch_price_history(tuple(tickers_used.values()), period)
        prices = prices.dropna()
        daily_returns = prices.pct_change().dropna()

        port_daily = pd.Series(0.0, index=daily_returns.index)
        for k, w in weights.items():
            ticker = TICKERS[k]
            if ticker is None:
                continue  # cash contributes via the risk-free adjustment below
            if ticker in daily_returns.columns:
                port_daily += w * daily_returns[ticker]

        cash_weight = weights.get("cash", 0.0)
        port_daily += cash_weight * (RISK_FREE_RATE / 252)

        ann_return = (1 + port_daily.mean()) ** 252 - 1
        ann_vol = port_daily.std() * np.sqrt(252)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else float("nan")

        cumulative = (1 + port_daily).cumprod() * 100

        # Benchmark: simple 60/40 (VFV.TO / XBB.TO)
        bench_tickers = ["VFV.TO", "XBB.TO"]
        bench_prices = fetch_price_history(tuple(bench_tickers), period).dropna()
        bench_returns = bench_prices.pct_change().dropna()
        bench_daily = 0.6 * bench_returns["VFV.TO"] + 0.4 * bench_returns["XBB.TO"]
        bench_cumulative = (1 + bench_daily).cumprod() * 100
        bench_ann_return = (1 + bench_daily.mean()) ** 252 - 1
        bench_ann_vol = bench_daily.std() * np.sqrt(252)
        bench_sharpe = (bench_ann_return - RISK_FREE_RATE) / bench_ann_vol if bench_ann_vol > 0 else float("nan")

        return {
            "cumulative": cumulative,
            "bench_cumulative": bench_cumulative,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "bench_ann_return": bench_ann_return,
            "bench_ann_vol": bench_ann_vol,
            "bench_sharpe": bench_sharpe,
        }
    except Exception as e:
        return {"error": str(e)}


# =======================================================================
# 4. NEWS (via yfinance's built-in news feed per ticker)
# =======================================================================
@st.cache_data(ttl=60 * 60 * 2)
def fetch_news_for_asset_classes():
    news_by_class = {}
    for key, ticker in TICKERS.items():
        if ticker is None:
            continue
        try:
            items = yf.Ticker(ticker).news[:3]
            headlines = []
            for item in items:
                content = item.get("content", item)  # yfinance news schema varies by version
                title = content.get("title") or item.get("title", "Untitled")
                link = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
                publisher = (content.get("provider") or {}).get("displayName", "")
                headlines.append({"title": title, "link": link, "publisher": publisher})
            news_by_class[key] = headlines
        except Exception:
            news_by_class[key] = []
    return news_by_class


# =======================================================================
# 5. SAVE / LOAD PORTFOLIOS
# =======================================================================
def load_saved_portfolios():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return []


def save_portfolio(name, amount, risk_profile, horizon, weights):
    portfolios = load_saved_portfolios()
    portfolios.append({
        "name": name,
        "date_saved": str(date.today()),
        "amount": amount,
        "risk_profile": risk_profile,
        "horizon": horizon,
        "weights": weights,
    })
    with open(SAVE_FILE, "w") as f:
        json.dump(portfolios, f, indent=2)


def delete_portfolio(index):
    portfolios = load_saved_portfolios()
    if 0 <= index < len(portfolios):
        portfolios.pop(index)
        with open(SAVE_FILE, "w") as f:
            json.dump(portfolios, f, indent=2)


# =======================================================================
# 6. PAGE CONFIG + STYLE
# =======================================================================
st.set_page_config(page_title="PortPicker", page_icon="📊", layout="wide")

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

st.title("📊 PortPicker")
st.caption("Builds a target asset allocation from your risk profile, horizon, and the prior trading day's market conditions.")

if "current_weights" not in st.session_state:
    st.session_state.current_weights = None
if "current_amount" not in st.session_state:
    st.session_state.current_amount = 20000.0
if "current_risk" not in st.session_state:
    st.session_state.current_risk = RiskProfile.NEUTRAL.value
if "current_horizon" not in st.session_state:
    st.session_state.current_horizon = 10

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🧮 Calculator", "📈 Backtest & Risk", "⚖️ Compare Profiles", "📰 Market News", "💾 Saved Portfolios", "📘 Strategy Guide"]
)


# =======================================================================
# TAB 1 — CALCULATOR
# =======================================================================
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Investment Amount ($)", min_value=1.0, value=st.session_state.current_amount, step=500.0)
    with col2:
        risk_choice = st.selectbox("Risk Profile", [p.value for p in RiskProfile],
                                    index=[p.value for p in RiskProfile].index(st.session_state.current_risk))
    with col3:
        horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=99,
                                   value=st.session_state.current_horizon, step=1)

    calculate = st.button("Calculate Portfolio")

    st.divider()

    score, as_of, vix_level = market_condition_score(str(date.today()))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Market Condition Score", f"{score:+.2f}" if score is not None else "N/A",
                   help="Range: -1 (risk-off) to +1 (risk-on)")
    with m2:
        st.metric("As-of Date (9:00 AM open)", as_of if score is not None else "unavailable")
    with m3:
        st.metric("Prior-Day VIX (open)", f"{vix_level:.2f}" if vix_level else "N/A")

    if calculate:
        risk_profile = RiskProfile(risk_choice)
        weights = build_portfolio(risk_profile, horizon, score)
        st.session_state.current_weights = weights
        st.session_state.current_amount = amount
        st.session_state.current_risk = risk_choice
        st.session_state.current_horizon = horizon

    if st.session_state.current_weights:
        weights = st.session_state.current_weights

        st.subheader("Recommended Allocation")
        left, right = st.columns([1, 1.3])
        with left:
            fig = go.Figure(data=[go.Pie(
                labels=[ASSET_LABELS[k] for k in weights],
                values=[v * 100 for v in weights.values()],
                hole=0.45, textinfo="label+percent",
            )])
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True, key="calc_pie")
        with right:
            df = pd.DataFrame({
                "Asset Class": [ASSET_LABELS[k] for k in weights],
                "Ticker": [TICKERS[k] or "—" for k in weights],
                "Weight": [f"{v*100:.1f}%" for v in weights.values()],
                "Dollar Amount": [f"${amount * v:,.2f}" for v in weights.values()],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total: ${amount:,.2f}**")

        st.divider()
        st.subheader("✏️ Manually Override Weights")
        st.caption("Adjust sliders below, then click 'Apply Manual Weights.' They'll auto-normalize to sum to 100%.")

        manual_weights = {}
        cols = st.columns(4)
        for i, k in enumerate(weights):
            with cols[i % 4]:
                manual_weights[k] = st.slider(ASSET_LABELS[k], 0, 100, int(round(weights[k] * 100)), key=f"slider_{k}")

        raw_total = sum(manual_weights.values())
        if raw_total > 0:
            normalized_manual = {k: v / raw_total for k, v in manual_weights.items()}
        else:
            normalized_manual = weights

        st.caption(f"Raw slider total: {raw_total}% (auto-normalized to 100% on apply)")

        if st.button("Apply Manual Weights"):
            st.session_state.current_weights = normalized_manual
            st.success("Manual weights applied — recommended allocation above is now overridden.")
            st.rerun()

        st.divider()
        st.subheader("💾 Save This Portfolio")
        save_name = st.text_input("Portfolio name", placeholder="e.g. My Growth Mix")
        if st.button("Save Portfolio"):
            if save_name.strip():
                save_portfolio(save_name.strip(), amount, risk_choice, horizon, weights)
                st.success(f"Saved '{save_name}' — view it under the Saved Portfolios tab.")
            else:
                st.warning("Give your portfolio a name first.")


# =======================================================================
# TAB 2 — BACKTEST & RISK
# =======================================================================
with tab2:
    st.subheader("Historical Backtest")
    if not st.session_state.current_weights:
        st.info("Calculate a portfolio on the Calculator tab first.")
    else:
        weights = st.session_state.current_weights
        bt_years = st.radio("Lookback period", [1, 5, 10], index=1, horizontal=True, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")

        if st.button("Run Backtest"):
            with st.spinner("Pulling historical data..."):
                result = run_backtest(weights, bt_years)

            if "error" in result:
                st.error(f"Backtest unavailable: {result['error']}")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["cumulative"].index, y=result["cumulative"],
                                          name="Your Portfolio", line=dict(color="#3498db", width=2)))
                fig.add_trace(go.Scatter(x=result["bench_cumulative"].index, y=result["bench_cumulative"],
                                          name="60/40 Benchmark", line=dict(color="#95a5a6", width=2, dash="dash")))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e8eaed"), legend=dict(orientation="h"),
                                   yaxis_title="Growth of $100")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Your Portfolio**")
                    st.metric("Annualized Return", f"{result['ann_return']*100:.2f}%")
                    st.metric("Annualized Volatility", f"{result['ann_vol']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
                with c2:
                    st.markdown("**60/40 Benchmark**")
                    st.metric("Annualized Return", f"{result['bench_ann_return']*100:.2f}%")
                    st.metric("Annualized Volatility", f"{result['bench_ann_vol']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{result['bench_sharpe']:.2f}")

                st.caption(f"Sharpe ratio assumes a {RISK_FREE_RATE*100:.1f}% annualized risk-free rate. Past performance is not indicative of future results.")


# =======================================================================
# TAB 3 — COMPARE PROFILES
# =======================================================================
with tab3:
    st.subheader("Compare All Risk Profiles")
    compare_amount = st.number_input("Amount for comparison ($)", min_value=1.0, value=20000.0, step=500.0, key="compare_amount")
    compare_horizon = st.number_input("Horizon for comparison (years)", min_value=1, max_value=99, value=10, key="compare_horizon")

    score, as_of, _ = market_condition_score(str(date.today()))

    cols = st.columns(3)
    for i, profile in enumerate(RiskProfile):
        w = build_portfolio(profile, compare_horizon, score)
        with cols[i]:
            st.markdown(f"**{profile.value}**")
            fig = go.Figure(data=[go.Pie(
                labels=[ASSET_LABELS[k] for k in w], values=[v * 100 for v in w.values()],
                hole=0.45, textinfo="percent",
            )])
            fig.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True, key=f"compare_pie_{i}")
            df = pd.DataFrame({
                "Asset": [ASSET_LABELS[k] for k in w],
                "Weight": [f"{v*100:.1f}%" for v in w.values()],
                "$": [f"${compare_amount*v:,.0f}" for v in w.values()],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)


# =======================================================================
# TAB 4 — MARKET NEWS
# =======================================================================
with tab4:
    st.subheader("Recent News by Asset Class")
    st.caption("Headlines pulled live per representative ticker. Click through to read the full article at the source.")

    news_by_class = fetch_news_for_asset_classes()
    for key, headlines in news_by_class.items():
        st.markdown(f"**{ASSET_LABELS[key]}** ({TICKERS[key]})")
        if not headlines:
            st.caption("No recent headlines available.")
        else:
            for h in headlines:
                pub = f" — *{h['publisher']}*" if h["publisher"] else ""
                if h["link"]:
                    st.markdown(f"- [{h['title']}]({h['link']}){pub}")
                else:
                    st.markdown(f"- {h['title']}{pub}")
        st.markdown("")


# =======================================================================
# TAB 5 — SAVED PORTFOLIOS
# =======================================================================
with tab5:
    st.subheader("Saved Portfolios")
    portfolios = load_saved_portfolios()

    if not portfolios:
        st.info("No saved portfolios yet — save one from the Calculator tab.")
    else:
        for i, p in enumerate(portfolios):
            with st.expander(f"**{p['name']}** — saved {p['date_saved']} ({p['risk_profile']}, {p['horizon']}yr, ${p['amount']:,.0f})"):
                w = p["weights"]
                df = pd.DataFrame({
                    "Asset": [ASSET_LABELS.get(k, k) for k in w],
                    "Weight": [f"{v*100:.1f}%" for v in w.values()],
                    "$": [f"${p['amount']*v:,.0f}" for v in w.values()],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.button("Delete", key=f"delete_{i}"):
                    delete_portfolio(i)
                    st.rerun()


# =======================================================================
# TAB 6 — STRATEGY GUIDE
# =======================================================================
with tab6:
    st.subheader("Strategy Breakdown by Risk Profile")

    for profile in RiskProfile:
        notes = STRATEGY_NOTES[profile]
        mix = STRATEGIC_MIX[profile]

        with st.expander(f"**{profile.value}** — {notes['summary']}", expanded=(profile == RiskProfile.NEUTRAL)):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                fig = go.Figure(data=[go.Pie(
                    labels=[ASSET_LABELS[k] for k in mix], values=[v * 100 for v in mix.values()],
                    hole=0.45, textinfo="label+percent",
                )])
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                                   paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"), height=280)
                st.plotly_chart(fig, use_container_width=True, key=f"strategy_pie_{profile.value}")
            with c2:
                st.markdown(f"**Philosophy:** {notes['philosophy']}")
                st.markdown(f"**Expected Volatility:** {notes['expected_volatility']}")
                st.markdown(f"**Best suited for:** {notes['who_its_for']}")

    st.divider()
    st.markdown("""
        **How the tactical tilt works:** Each profile's weights above are the *strategic* baseline.
        On the Calculator tab, two things nudge the final allocation within a ±15% range:
        - **Horizon** — longer horizons tilt toward equities/REITs, shorter horizons tilt toward safety
        - **Market conditions** — calculated from the previous trading day's VIX level and
          S&P 500 momentum at market open
    """)
