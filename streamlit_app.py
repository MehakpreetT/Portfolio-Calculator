"""
PortPicker — Forward-Looking Portfolio Construction Web App
--------------------------------------------------------------
Run locally with:  streamlit run streamlit_app.py

Login: first run auto-creates a default account file (users.yaml) with
a seeded test account (username: testuser, password: test1234).
Register a new account from the Register tab shown before you log in.

Pages (left sidebar once logged in):
  1. Calculator          -> build a portfolio, add custom tickers,
                            override weights, set a stop-loss, validate,
                            export, and save it to your account
  2. Backtest & Risk      -> historical performance + Sharpe ratio
  3. Efficient Frontier   -> Monte Carlo frontier for YOUR current holdings,
                            with your actual portfolio plotted on it
  4. Portfolio Education  -> what each asset class is and why it's used
  5. Compare Profiles     -> Conservative / Neutral / Growth side-by-side
  6. Market News          -> recent headlines per asset class
  7. Saved Portfolios     -> your saved strategies (per account)
  8. Strategy Guide       -> philosophy behind each risk profile
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from fpdf import FPDF
from scipy.optimize import minimize
from enum import Enum
from datetime import date

USERS_FILE = "users.yaml"
SAVE_DIR = "saved_portfolios"
RISK_FREE_RATE = 0.03

os.makedirs(SAVE_DIR, exist_ok=True)


# =======================================================================
# 1. CONFIG / CONSTANTS
# =======================================================================
class RiskProfile(Enum):
    CONSERVATIVE = "Conservative"
    NEUTRAL = "Neutral"
    GROWTH = "Growth"


STRATEGIC_MIX = {
    RiskProfile.CONSERVATIVE: {"cash": 0.02, "bonds": 0.53, "cdn_eq": 0.12, "us_eq": 0.13, "intl_eq": 0.10, "em_eq": 0.00, "gold": 0.05, "reit": 0.05},
    RiskProfile.NEUTRAL:      {"cash": 0.02, "bonds": 0.33, "cdn_eq": 0.13, "us_eq": 0.22, "intl_eq": 0.13, "em_eq": 0.04, "gold": 0.05, "reit": 0.08},
    RiskProfile.GROWTH:       {"cash": 0.02, "bonds": 0.18, "cdn_eq": 0.15, "us_eq": 0.26, "intl_eq": 0.16, "em_eq": 0.07, "gold": 0.06, "reit": 0.10},
}

TICKERS = {
    "cash": None, "bonds": "XBB.TO", "cdn_eq": "XIC.TO", "us_eq": "VFV.TO",
    "intl_eq": "XEF.TO", "em_eq": "XEC.TO", "gold": "CGL.TO", "reit": "XRE.TO",
}

ASSET_LABELS = {
    "cash": "Cash", "bonds": "Fixed Income", "cdn_eq": "Canadian Equity", "us_eq": "U.S. Equity",
    "intl_eq": "International Equity", "em_eq": "Emerging Markets", "gold": "Gold / Commodities", "reit": "REITs",
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

EDUCATION_CONTENT = {
    "cash": {
        "what": "Cash and cash-equivalents (like money market funds) are the most stable, liquid holdings in a portfolio — the closest thing to zero price risk.",
        "why": "Cash cushions the portfolio during downturns and provides dry powder to rebalance into other assets when they get cheap. It typically earns the least over time, so portfolios only hold a small amount.",
    },
    "bonds": {
        "what": "Fixed income (bonds) are loans to governments or corporations that pay a set interest rate over a defined term.",
        "why": "Bonds are less volatile than stocks and often move differently than equities, especially during stock market downturns — they're the main shock absorber in a balanced portfolio.",
    },
    "cdn_eq": {
        "what": "Canadian equities are shares of Canadian companies, often concentrated in financials, energy, and materials.",
        "why": "Gives home-market exposure and dividend income, though Canada's market is more concentrated in a few sectors than global markets.",
    },
    "us_eq": {
        "what": "U.S. equities are shares of American companies, spanning the world's largest and most liquid stock market.",
        "why": "The U.S. market offers the broadest sector diversification (especially technology and healthcare) and has historically been a strong long-term growth driver.",
    },
    "intl_eq": {
        "what": "International (developed market) equities are shares of companies outside North America — mainly Europe, Japan, and Australia.",
        "why": "Adds geographic diversification so the portfolio isn't dependent on any single country's economic cycle.",
    },
    "em_eq": {
        "what": "Emerging market equities are shares of companies in developing economies like China, India, and Brazil.",
        "why": "Offers higher long-run growth potential from faster-growing economies, at the cost of higher volatility and political/currency risk.",
    },
    "gold": {
        "what": "Gold and broader commodities are physical/real assets rather than claims on a company's earnings.",
        "why": "Gold tends to hold value during inflation or crisis periods when stocks and bonds can both struggle, making it a useful diversifier.",
    },
    "reit": {
        "what": "REITs (Real Estate Investment Trusts) are companies that own and operate income-producing real estate, traded like stocks.",
        "why": "Gives real estate exposure and steady income without directly owning property, and often behaves differently than the broader stock market.",
    },
}


# =======================================================================
# 2. USER ACCOUNTS
# =======================================================================
def load_or_create_user_config():
    cookie_key = st.secrets.get("cookie_key", None) if hasattr(st, "secrets") else None
    if cookie_key is None:
        st.error("No cookie_key found in st.secrets — see the setup instructions to add one before using login.")
        st.stop()

    if not os.path.exists(USERS_FILE):
        config = {
            "credentials": {"usernames": {}},
            "cookie": {"name": "portpicker_auth", "key": cookie_key, "expiry_days": 30},
        }
        with open(USERS_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    with open(USERS_FILE, "r") as f:
        loaded = yaml.load(f, Loader=SafeLoader)
    loaded["cookie"]["key"] = cookie_key

    if "testuser" not in loaded["credentials"]["usernames"]:
        loaded["credentials"]["usernames"]["testuser"] = {
            "email": "test@portpicker.demo", "name": "Test User",
            "password": stauth.Hasher().hash("test1234"),
        }
        with open(USERS_FILE, "w") as f:
            yaml.dump(loaded, f, default_flow_style=False)

    return loaded


def save_user_config(config):
    with open(USERS_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def user_portfolio_file(username):
    return os.path.join(SAVE_DIR, f"{username}.json")


def load_saved_portfolios(username):
    path = user_portfolio_file(username)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_portfolio(username, name, amount, risk_profile, horizon, weights, stop_loss=None):
    portfolios = load_saved_portfolios(username)
    portfolios.append({
        "name": name, "date_saved": str(date.today()), "amount": amount,
        "risk_profile": risk_profile, "horizon": horizon, "weights": weights, "stop_loss": stop_loss,
    })
    with open(user_portfolio_file(username), "w") as f:
        json.dump(portfolios, f, indent=2)


def delete_portfolio(username, index):
    portfolios = load_saved_portfolios(username)
    if 0 <= index < len(portfolios):
        portfolios.pop(index)
        with open(user_portfolio_file(username), "w") as f:
            json.dump(portfolios, f, indent=2)


# =======================================================================
# 3. CORE ALLOCATION LOGIC
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
    return {k: v / total for k, v in adjusted.items()}


# =======================================================================
# 4. CUSTOM TICKER VALIDATION
# =======================================================================
@st.cache_data(ttl=60 * 60 * 6)
def validate_ticker(ticker: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return False, None
        info = t.info
        name = info.get("shortName") or info.get("longName") or ticker
        return True, name
    except Exception:
        return False, None


# =======================================================================
# 5. PRICE HISTORY / BACKTEST / EFFICIENT FRONTIER
# =======================================================================
@st.cache_data(ttl=60 * 60 * 6)
def fetch_price_history(tickers_tuple, period):
    tickers = [t for t in tickers_tuple if t is not None]
    data = yf.download(tickers, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


def run_backtest(weights: dict, tickers_map: dict, years: int, stop_loss_pct=None):
    period = "1y" if years <= 1 else ("5y" if years <= 5 else "10y")
    tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}

    try:
        prices = fetch_price_history(tuple(tickers_used.values()), period)
        prices = prices.dropna()
        daily_returns = prices.pct_change().dropna()

        port_daily = pd.Series(0.0, index=daily_returns.index)
        for k, w in weights.items():
            ticker = tickers_map.get(k)
            if ticker is None:
                continue
            if ticker in daily_returns.columns:
                port_daily += w * daily_returns[ticker]

        cash_weight = weights.get("cash", 0.0)
        port_daily += cash_weight * (RISK_FREE_RATE / 252)

        ann_return = (1 + port_daily.mean()) ** 252 - 1
        ann_vol = port_daily.std() * np.sqrt(252)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else float("nan")

        cumulative = (1 + port_daily).cumprod() * 100
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        stop_loss_breach_date = None
        if stop_loss_pct is not None:
            breached = drawdown[drawdown <= -abs(stop_loss_pct) / 100]
            if not breached.empty:
                stop_loss_breach_date = breached.index[0].strftime("%Y-%m-%d")

        bench_tickers = ["VFV.TO", "XBB.TO"]
        bench_prices = fetch_price_history(tuple(bench_tickers), period).dropna()
        bench_returns = bench_prices.pct_change().dropna()
        bench_daily = 0.6 * bench_returns["VFV.TO"] + 0.4 * bench_returns["XBB.TO"]
        bench_cumulative = (1 + bench_daily).cumprod() * 100
        bench_ann_return = (1 + bench_daily.mean()) ** 252 - 1
        bench_ann_vol = bench_daily.std() * np.sqrt(252)
        bench_sharpe = (bench_ann_return - RISK_FREE_RATE) / bench_ann_vol if bench_ann_vol > 0 else float("nan")

        return {
            "cumulative": cumulative, "bench_cumulative": bench_cumulative,
            "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": max_drawdown, "stop_loss_breach_date": stop_loss_breach_date,
            "bench_ann_return": bench_ann_return, "bench_ann_vol": bench_ann_vol, "bench_sharpe": bench_sharpe,
            "daily_returns": daily_returns, "tickers_used": tickers_used, "weights_used": weights,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_efficient_frontier(daily_returns: pd.DataFrame, weights: dict, tickers_map: dict, n_portfolios=3000):
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    n_assets = len(mean_returns)
    tickers_order = mean_returns.index.tolist()

    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = sharpe

    def neg_sharpe(w):
        r = np.dot(w, mean_returns)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(r - RISK_FREE_RATE) / v if v > 0 else 0

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1 / n_assets] * n_assets)
    opt_result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_weights = opt_result.x
    opt_return = np.dot(optimal_weights, mean_returns)
    opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    # Map your ACTUAL held weights (by asset class) onto the ticker-ordered vector
    # so we can plot exactly where your current portfolio sits on this same frontier.
    your_weight_vec = np.zeros(n_assets)
    for k, w in weights.items():
        ticker = tickers_map.get(k)
        if ticker in tickers_order:
            idx = tickers_order.index(ticker)
            your_weight_vec[idx] += w
    non_cash_total = your_weight_vec.sum()
    cash_weight = weights.get("cash", 0.0)
    if non_cash_total > 0:
        your_return = np.dot(your_weight_vec, mean_returns) + cash_weight * RISK_FREE_RATE
        your_vol = np.sqrt(np.dot(your_weight_vec.T, np.dot(cov_matrix, your_weight_vec)))
    else:
        your_return, your_vol = None, None

    return results, tickers_order, optimal_weights, opt_return, opt_vol, your_return, your_vol


# =======================================================================
# 6. NEWS
# =======================================================================
@st.cache_data(ttl=60 * 60 * 2)
def fetch_news_for_asset_classes(tickers_map):
    news_by_class = {}
    for key, ticker in tickers_map.items():
        if ticker is None:
            continue
        try:
            items = yf.Ticker(ticker).news[:3]
            headlines = []
            for item in items:
                content = item.get("content", item)
                title = content.get("title") or item.get("title", "Untitled")
                link = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
                publisher = (content.get("provider") or {}).get("displayName", "")
                headlines.append({"title": title, "link": link, "publisher": publisher})
            news_by_class[key] = headlines
        except Exception:
            news_by_class[key] = []
    return news_by_class


# =======================================================================
# 7. EXPORT (CSV + PDF)
# =======================================================================
def build_csv(weights, amount, labels_map, tickers_map):
    df = pd.DataFrame({
        "Asset Class": [labels_map.get(k, k) for k in weights],
        "Ticker": [tickers_map.get(k) or "N/A" for k in weights],
        "Weight (%)": [round(v * 100, 2) for v in weights.values()],
        "Dollar Amount": [round(amount * v, 2) for v in weights.values()],
    })
    return df.to_csv(index=False).encode("utf-8")


def build_pdf(weights, amount, risk_profile, horizon, labels_map, tickers_map):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PortPicker - Portfolio Allocation Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {date.today()}", ln=True)
    pdf.cell(0, 8, f"Amount: ${amount:,.2f}   Risk Profile: {risk_profile}   Horizon: {horizon} yrs", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(70, 8, "Asset Class", border=1)
    pdf.cell(40, 8, "Ticker", border=1)
    pdf.cell(30, 8, "Weight", border=1)
    pdf.cell(40, 8, "Dollar Amount", border=1, ln=True)

    pdf.set_font("Helvetica", "", 10)
    for k, w in weights.items():
        pdf.cell(70, 8, labels_map.get(k, k), border=1)
        pdf.cell(40, 8, str(tickers_map.get(k) or "N/A"), border=1)
        pdf.cell(30, 8, f"{w*100:.1f}%", border=1)
        pdf.cell(40, 8, f"${amount*w:,.2f}", border=1, ln=True)

    return bytes(pdf.output(dest="S"))


# =======================================================================
# 8. PAGE CONFIG + STYLE
# =======================================================================
st.set_page_config(page_title="PortPicker", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        h1, h2, h3 { color: #e8eaed; letter-spacing: -0.3px; }
        [data-testid="stMetricValue"] { color: #e8eaed; }
        [data-testid="stMetric"] {
            background-color: #171b24; border: 1px solid #2b2f38;
            border-radius: 10px; padding: 12px 16px;
        }
        .stButton>button {
            background-color: #2563eb; color: white; border: none; border-radius: 8px;
            font-weight: 600; padding: 0.5em 1.5em; transition: background-color 0.15s ease;
        }
        .stButton>button:hover { background-color: #1d4ed8; color: white; }
        .stDownloadButton>button {
            background-color: #1f2937; color: white; border: 1px solid #374151; border-radius: 8px;
        }
        section[data-testid="stSidebar"] { background-color: #10131a; border-right: 1px solid #232733; }
        .edu-card {
            background-color: #171b24; border: 1px solid #2b2f38; border-radius: 10px;
            padding: 16px 20px; margin-bottom: 14px;
        }
        .edu-card h4 { margin: 0 0 8px 0; color: #60a5fa; }
    </style>
""", unsafe_allow_html=True)

config = load_or_create_user_config()
authenticator = stauth.Authenticate(
    config["credentials"], config["cookie"]["name"], config["cookie"]["key"], config["cookie"]["expiry_days"]
)

# --- Login gate: only show login/register UI when NOT authenticated ---
if not st.session_state.get("authentication_status"):
    st.title("📊 PortPicker")
    st.caption("Log in or create an account to build and save portfolios.")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        authenticator.login()
        if st.session_state.get("authentication_status") is False:
            st.error("Username or password is incorrect.")
    with register_tab:
        try:
            email, username, name = authenticator.register_user(pre_authorized=None)
            if email:
                save_user_config(config)
                st.success("Account created — please log in from the Login tab.")
        except Exception as e:
            st.error(str(e))
    st.stop()

# =======================================================================
# LOGGED IN FROM HERE ON — login/register UI is fully hidden
# =======================================================================
username = st.session_state["username"]

for key, default in [
    ("current_weights", None), ("current_amount", 20000.0), ("current_risk", RiskProfile.NEUTRAL.value),
    ("current_horizon", 10), ("custom_tickers", {}), ("stop_loss_pct", 15.0), ("is_customized", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown(f"### 📊 PortPicker")
    st.write(f"Logged in as **{st.session_state['name']}**")
    authenticator.logout()
    st.divider()
    page = option_menu(
        menu_title=None,
        options=["Calculator", "Backtest & Risk", "Efficient Frontier", "Portfolio Education",
                 "Compare Profiles", "Market News", "Saved Portfolios", "Strategy Guide"],
        icons=["calculator", "graph-up-arrow", "bullseye", "mortarboard",
               "bar-chart-steps", "newspaper", "save", "book"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#10131a"},
            "icon": {"color": "#60a5fa", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "color": "#c9ccd3", "--hover-color": "#1a1e29"},
            "nav-link-selected": {"background-color": "#1d4ed8"},
        },
    )

st.caption("Builds a target asset allocation from your risk profile, horizon, and the prior trading day's market conditions.")


# =======================================================================
# PAGE: CALCULATOR
# =======================================================================
if page == "Calculator":
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
        st.metric("Market Condition Score", f"{score:+.2f}" if score is not None else "N/A")
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
        st.session_state.is_customized = False

    if st.session_state.current_weights:
        weights = dict(st.session_state.current_weights)

        st.divider()
        st.subheader("➕ Add a Custom Ticker")
        st.caption("Search any North American ticker (e.g. AAPL, SHOP.TO, TSLA) to add it as its own holding.")
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            custom_ticker_input = st.text_input("Ticker symbol", placeholder="e.g. AAPL", key="custom_ticker_box")
        with cc2:
            add_clicked = st.button("Add Ticker")
        if add_clicked:
            cleaned = (custom_ticker_input or "").strip().upper()
            if not cleaned:
                st.warning("Type a ticker symbol first.")
            else:
                is_valid, display_name = validate_ticker(cleaned)
                if is_valid:
                    key = f"custom_{cleaned}"
                    st.session_state.custom_tickers[key] = {"ticker": cleaned, "name": display_name}
                    st.session_state.current_weights[key] = 0.05
                    st.session_state.is_customized = True
                    st.success(f"Added {display_name} ({cleaned})")
                    st.rerun()
                else:
                    st.error(f"Couldn't validate ticker '{cleaned}' — check the symbol and try again.")

        if st.session_state.custom_tickers:
            st.caption("Custom tickers added: " + ", ".join(v["ticker"] for v in st.session_state.custom_tickers.values()))
            if st.button("Clear custom tickers"):
                # Remove custom keys from BOTH custom_tickers and current_weights so
                # nothing downstream tries to look up a label/ticker that no longer exists.
                removed_keys = list(st.session_state.custom_tickers.keys())
                st.session_state.custom_tickers = {}
                if st.session_state.current_weights:
                    for k in removed_keys:
                        st.session_state.current_weights.pop(k, None)
                    total = sum(st.session_state.current_weights.values())
                    if total > 0:
                        st.session_state.current_weights = {k: v / total for k, v in st.session_state.current_weights.items()}
                st.session_state.is_customized = True
                st.rerun()

        # Build combined tickers/labels map (base + custom) — use dict copies so a
        # ticker removed from custom_tickers can never leave a stale label lookup.
        combined_tickers = dict(TICKERS)
        combined_labels = dict(ASSET_LABELS)
        for key, v in st.session_state.custom_tickers.items():
            combined_tickers[key] = v["ticker"]
            combined_labels[key] = f"{v['name']} ({v['ticker']})"

        # Defensive filter: drop any weight key that doesn't resolve to a known
        # label (guards against any future desync between the two dicts).
        weights = {k: v for k, v in weights.items() if k in combined_labels}

        st.divider()
        heading = "Your Custom Allocation" if st.session_state.is_customized else "Recommended Allocation"
        st.subheader(heading)
        left, right = st.columns([1, 1.3])
        with left:
            fig = go.Figure(data=[go.Pie(
                labels=[combined_labels[k] for k in weights], values=[v * 100 for v in weights.values()],
                hole=0.45, textinfo="label+percent",
            )])
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True, key="calc_pie")
        with right:
            df = pd.DataFrame({
                "Asset Class": [combined_labels[k] for k in weights],
                "Ticker": [combined_tickers.get(k) or "—" for k in weights],
                "Weight": [f"{v*100:.1f}%" for v in weights.values()],
                "Dollar Amount": [f"${amount * v:,.2f}" for v in weights.values()],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total: ${amount:,.2f}**")

        st.divider()
        st.subheader("✏️ Manually Override Weights")
        manual_weights = {}
        cols = st.columns(4)
        for i, k in enumerate(weights):
            with cols[i % 4]:
                manual_weights[k] = st.slider(combined_labels[k], 0, 100, int(round(weights[k] * 100)), key=f"slider_{k}")
        raw_total = sum(manual_weights.values())
        normalized_manual = {k: v / raw_total for k, v in manual_weights.items()} if raw_total > 0 else weights
        st.caption(f"Raw slider total: {raw_total}% (auto-normalized to 100% on apply)")
        if st.button("Apply Manual Weights"):
            st.session_state.current_weights = normalized_manual
            st.session_state.is_customized = True
            st.success("Manual weights applied.")
            st.rerun()

        st.divider()
        st.subheader("✅ Data Validation")
        issues = []
        if abs(sum(weights.values()) - 1.0) > 0.01:
            issues.append(f"Weights sum to {sum(weights.values())*100:.1f}%, not 100%.")
        if amount <= 0:
            issues.append("Investment amount must be greater than zero.")
        for k, t in combined_tickers.items():
            if t is not None and k in weights and weights[k] > 0:
                is_valid, _ = validate_ticker(t)
                if not is_valid:
                    issues.append(f"Ticker '{t}' could not be validated — data may be unavailable.")
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("All checks passed — weights sum to 100%, amount is valid, and all tickers resolved.")

        st.divider()
        st.subheader("🛑 Stop-Loss Threshold")
        stop_loss_pct = st.number_input("Stop-loss (% decline from peak value)", min_value=0.0, max_value=100.0,
                                         value=st.session_state.stop_loss_pct, step=1.0)
        st.session_state.stop_loss_pct = stop_loss_pct
        st.caption("Used on the Backtest tab to flag whether this threshold would have been breached historically.")

        st.divider()
        st.subheader("📤 Export")
        exp1, exp2 = st.columns(2)
        with exp1:
            csv_bytes = build_csv(weights, amount, combined_labels, combined_tickers)
            st.download_button("Download CSV", csv_bytes, file_name="portpicker_allocation.csv", mime="text/csv")
        with exp2:
            try:
                pdf_bytes = build_pdf(weights, amount, risk_choice, horizon, combined_labels, combined_tickers)
                st.download_button("Download PDF", pdf_bytes, file_name="portpicker_allocation.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF export failed: {e}")

        st.divider()
        st.subheader("💾 Save This Portfolio")
        save_name = st.text_input("Portfolio name", placeholder="e.g. My Growth Mix")
        if st.button("Save Portfolio"):
            if save_name.strip():
                save_portfolio(username, save_name.strip(), amount, risk_choice, horizon, weights, stop_loss_pct)
                st.success(f"Saved '{save_name}' to your account.")
            else:
                st.warning("Give your portfolio a name first.")

        st.session_state["_combined_tickers"] = combined_tickers
        st.session_state["_combined_labels"] = combined_labels
        st.session_state["current_weights"] = weights


# =======================================================================
# PAGE: BACKTEST & RISK
# =======================================================================
elif page == "Backtest & Risk":
    st.subheader("Historical Backtest")
    if not st.session_state.current_weights:
        st.info("Calculate a portfolio on the Calculator page first.")
    else:
        weights = st.session_state.current_weights
        tickers_map = st.session_state.get("_combined_tickers", TICKERS)
        bt_years = st.radio("Lookback period", [1, 5, 10], index=1, horizontal=True, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")

        if st.button("Run Backtest"):
            with st.spinner("Pulling historical data..."):
                result = run_backtest(weights, tickers_map, bt_years, st.session_state.stop_loss_pct)
            st.session_state["_last_backtest"] = result

        result = st.session_state.get("_last_backtest")
        if result:
            if "error" in result:
                st.error(f"Backtest unavailable: {result['error']}")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["cumulative"].index, y=result["cumulative"],
                                          name="Your Portfolio", line=dict(color="#3498db", width=2)))
                fig.add_trace(go.Scatter(x=result["bench_cumulative"].index, y=result["bench_cumulative"],
                                          name="60/40 Benchmark", line=dict(color="#95a5a6", width=2, dash="dash")))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e8eaed"), legend=dict(orientation="h"), yaxis_title="Growth of $100")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Your Portfolio**")
                    st.metric("Annualized Return", f"{result['ann_return']*100:.2f}%")
                    st.metric("Annualized Volatility", f"{result['ann_vol']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
                    st.metric("Max Drawdown", f"{result['max_drawdown']*100:.2f}%")
                with c2:
                    st.markdown("**60/40 Benchmark**")
                    st.metric("Annualized Return", f"{result['bench_ann_return']*100:.2f}%")
                    st.metric("Annualized Volatility", f"{result['bench_ann_vol']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{result['bench_sharpe']:.2f}")

                st.divider()
                if result["stop_loss_breach_date"]:
                    st.error(f"⚠️ Your stop-loss threshold of {st.session_state.stop_loss_pct:.0f}% would have been breached on {result['stop_loss_breach_date']} (max drawdown over this period: {result['max_drawdown']*100:.1f}%).")
                else:
                    st.success(f"✅ Your stop-loss threshold of {st.session_state.stop_loss_pct:.0f}% was not breached over this period (max drawdown: {result['max_drawdown']*100:.1f}%).")

                st.caption(f"Sharpe ratio assumes a {RISK_FREE_RATE*100:.1f}% annualized risk-free rate. Past performance is not indicative of future results.")


# =======================================================================
# PAGE: EFFICIENT FRONTIER
# =======================================================================
elif page == "Efficient Frontier":
    st.subheader("Efficient Frontier (Monte Carlo)")

    if not st.session_state.current_weights:
        st.info("Calculate a portfolio on the Calculator page first.")
    else:
        weights = st.session_state.current_weights
        tickers_map = st.session_state.get("_combined_tickers", TICKERS)
        labels_map = st.session_state.get("_combined_labels", ASSET_LABELS)
        held_names = [labels_map.get(k, k) for k, w in weights.items() if w > 0]

        fr_years = st.radio("Lookback period", [1, 5, 10], index=1, horizontal=True,
                             format_func=lambda y: f"{y} year{'s' if y > 1 else ''}", key="frontier_years")

        st.caption(
            f"Frontier for: **{st.session_state.current_risk} profile, ${st.session_state.current_amount:,.0f}, "
            f"{st.session_state.current_horizon}yr horizon** — simulated using your currently held assets "
            f"({', '.join(held_names)}) over the selected lookback period. This always reflects whatever "
            f"portfolio is currently active on the Calculator page, including manual overrides and custom tickers. "
            f"3,000 random-weight portfolios are simulated across these assets; cash is excluded from the "
            f"simulation itself (it has no price series) but is folded into the risk-free contribution."
        )

        # Fingerprint the current portfolio so a stale frontier from a different
        # portfolio/lookback is never shown as if it were current.
        portfolio_fingerprint = (tuple(sorted(weights.items())), tuple(sorted((k, v) for k, v in tickers_map.items())), fr_years)

        if st.button("Generate Efficient Frontier"):
            period = "1y" if fr_years <= 1 else ("5y" if fr_years <= 5 else "10y")
            tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}
            try:
                with st.spinner("Pulling price history and simulating portfolios..."):
                    prices = fetch_price_history(tuple(tickers_used.values()), period)
                    daily_returns = prices.dropna().pct_change().dropna()
                    sim_results, tickers_order, optimal_weights, opt_return, opt_vol, your_return, your_vol = \
                        compute_efficient_frontier(daily_returns, weights, tickers_map)
                    st.session_state["_frontier"] = (
                        sim_results, tickers_order, optimal_weights, opt_return, opt_vol,
                        your_return, your_vol, portfolio_fingerprint
                    )
            except Exception as e:
                st.error(f"Couldn't generate the frontier: {e}")

        cached = st.session_state.get("_frontier")
        if cached:
            sim_results, tickers_order, optimal_weights, opt_return, opt_vol, your_return, your_vol, cached_fingerprint = cached

            if cached_fingerprint != portfolio_fingerprint:
                st.warning("Your portfolio has changed since this frontier was generated — click 'Generate Efficient Frontier' again to refresh it.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sim_results[0], y=sim_results[1], mode="markers",
                marker=dict(size=4, color=sim_results[2], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
                name="Simulated Portfolios",
            ))
            fig.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_return], mode="markers",
                marker=dict(size=16, color="red", symbol="star"), name="Max-Sharpe Portfolio",
            ))
            if your_return is not None:
                fig.add_trace(go.Scatter(
                    x=[your_vol], y=[your_return], mode="markers",
                    marker=dict(size=16, color="#e8eaed", symbol="diamond", line=dict(color="black", width=1)),
                    name="Your Current Portfolio",
                ))
            fig.update_layout(
                xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            if your_return is not None:
                st.info(f"**Your Current Portfolio** sits at {your_vol*100:.2f}% volatility / {your_return*100:.2f}% return — "
                        f"the white diamond on the chart above.")

            st.markdown("**Max-Sharpe Portfolio Weights** (ticker-level, unconstrained by your risk profile)")
            opt_df = pd.DataFrame({"Ticker": tickers_order, "Weight": [f"{w*100:.1f}%" for w in optimal_weights]})
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
            st.caption(f"Annualized Return: {opt_return*100:.2f}%  |  Annualized Volatility: {opt_vol*100:.2f}%")


# =======================================================================
# PAGE: PORTFOLIO EDUCATION
# =======================================================================
elif page == "Portfolio Education":
    st.subheader("How Portfolio Construction Works")
    st.markdown("""
    Building a portfolio comes down to five steps:

    1. **Set an objective** — what's the money for, and when do you need it? This determines how much risk you can afford to take.
    2. **Choose an asset allocation** — split your money across asset classes (stocks, bonds, cash, real assets) based on that objective.
    3. **Pick the actual holdings** — select specific funds, ETFs, or securities to fill each asset class.
    4. **Monitor and rebalance** — as markets move, your weights drift from target. Rebalancing means trimming what's grown and adding to what's lagged, to stay aligned with your original plan.
    5. **Review the objective periodically** — your risk tolerance and time horizon change over time (e.g. getting closer to retirement), so the allocation should evolve too.
    """)

    st.divider()
    st.subheader("Asset Classes Explained")
    st.caption("What each holding in your portfolio actually is, and why it's included.")

    for key, content in EDUCATION_CONTENT.items():
        st.markdown(f"""
            <div class="edu-card">
                <h4>{ASSET_LABELS[key]}</h4>
                <p><strong>What it is:</strong> {content['what']}</p>
                <p><strong>Why it's used:</strong> {content['why']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Why Diversify Across All of These?")
    st.markdown("""
    No single asset class performs best every year, and different asset classes often respond differently
    to the same economic event (e.g. bonds can rise when stocks fall). Holding a mix smooths out the ride —
    you give up some upside in the best years in exchange for far less pain in the worst ones.
    """)


# =======================================================================
# PAGE: COMPARE PROFILES
# =======================================================================
elif page == "Compare Profiles":
    st.subheader("Compare All Risk Profiles")
    compare_amount = st.number_input("Amount for comparison ($)", min_value=1.0, value=20000.0, step=500.0, key="compare_amount")
    compare_horizon = st.number_input("Horizon for comparison (years)", min_value=1, max_value=99, value=10, key="compare_horizon")

    score, as_of, _ = market_condition_score(str(date.today()))
    cols = st.columns(3)
    for i, profile in enumerate(RiskProfile):
        w = build_portfolio(profile, compare_horizon, score)
        with cols[i]:
            st.markdown(f"**{profile.value}**")
            fig = go.Figure(data=[go.Pie(labels=[ASSET_LABELS[k] for k in w], values=[v * 100 for v in w.values()],
                                          hole=0.45, textinfo="percent")])
            fig.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True, key=f"compare_pie_{i}")
            df = pd.DataFrame({"Asset": [ASSET_LABELS[k] for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()],
                                "$": [f"${compare_amount*v:,.0f}" for v in w.values()]})
            st.dataframe(df, use_container_width=True, hide_index=True)


# =======================================================================
# PAGE: MARKET NEWS
# =======================================================================
elif page == "Market News":
    st.subheader("Recent News by Asset Class")
    tickers_map = st.session_state.get("_combined_tickers", TICKERS)
    labels_map = st.session_state.get("_combined_labels", ASSET_LABELS)
    news_by_class = fetch_news_for_asset_classes(tickers_map)
    for key, headlines in news_by_class.items():
        st.markdown(f"**{labels_map.get(key,key)}** ({tickers_map[key]})")
        if not headlines:
            st.caption("No recent headlines available.")
        else:
            for h in headlines:
                pub = f" — *{h['publisher']}*" if h["publisher"] else ""
                st.markdown(f"- [{h['title']}]({h['link']}){pub}" if h["link"] else f"- {h['title']}{pub}")
        st.markdown("")


# =======================================================================
# PAGE: SAVED PORTFOLIOS
# =======================================================================
elif page == "Saved Portfolios":
    st.subheader("Your Saved Portfolios")
    portfolios = load_saved_portfolios(username)
    if not portfolios:
        st.info("No saved portfolios yet — save one from the Calculator page.")
    else:
        for i, p in enumerate(portfolios):
            sl = f", stop-loss {p['stop_loss']:.0f}%" if p.get("stop_loss") else ""
            with st.expander(f"**{p['name']}** — saved {p['date_saved']} ({p['risk_profile']}, {p['horizon']}yr, ${p['amount']:,.0f}{sl})"):
                w = p["weights"]
                df = pd.DataFrame({"Asset": [ASSET_LABELS.get(k, k) for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()],
                                    "$": [f"${p['amount']*v:,.0f}" for v in w.values()]})
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.button("Delete", key=f"delete_{i}"):
                    delete_portfolio(username, i)
                    st.rerun()


# =======================================================================
# PAGE: STRATEGY GUIDE
# =======================================================================
elif page == "Strategy Guide":
    st.subheader("Strategy Breakdown by Risk Profile")
    for profile in RiskProfile:
        notes = STRATEGY_NOTES[profile]
        mix = STRATEGIC_MIX[profile]
        with st.expander(f"**{profile.value}** — {notes['summary']}", expanded=(profile == RiskProfile.NEUTRAL)):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                fig = go.Figure(data=[go.Pie(labels=[ASSET_LABELS[k] for k in mix], values=[v * 100 for v in mix.values()],
                                              hole=0.45, textinfo="label+percent")])
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
        On the Calculator page, two things nudge the final allocation within a ±15% range:
        - **Horizon** — longer horizons tilt toward equities/REITs, shorter horizons tilt toward safety
        - **Market conditions** — calculated from the previous trading day's VIX level and S&P 500 momentum at market open
    """)
