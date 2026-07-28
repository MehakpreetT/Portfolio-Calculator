"""
Wealthscope — Forward-Looking Portfolio Construction Web App
--------------------------------------------------------------
Run locally with:  streamlit run streamlit_app.py

Login: first run auto-creates a default account file (users.yaml) with
a seeded test account (username: testuser, password: test1234).

Pages (left sidebar once logged in):
  Dashboard, Risk Questionnaire, Calculator, Backtest & Risk,
  Stress Testing, Efficient Frontier, Sensitivity Index,
  Rebalancing Simulator, Quarterly Views, Compare Profiles,
  Market News, Saved Portfolios, Portfolio Education, Strategy Guide
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
from datetime import date, timedelta

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

# Illustrative long-run capital market assumptions (annualized), used only to show
# an indicative expected-return figure that shifts with horizon/tilt — NOT a forecast.
EXPECTED_RETURNS = {
    "cash": 0.03, "bonds": 0.04, "cdn_eq": 0.07, "us_eq": 0.085,
    "intl_eq": 0.065, "em_eq": 0.09, "gold": 0.05, "reit": 0.07,
}

TACTICAL_RANGE = 0.15

STRATEGY_NOTES = {
    RiskProfile.CONSERVATIVE: {
        "summary": "Prioritizes capital preservation with modest growth.",
        "philosophy": "The majority of the portfolio sits in fixed income to dampen volatility, with small equity, gold, and REIT sleeves for diversification and inflation protection. Emerging markets are excluded to avoid the sharpest drawdowns.",
        "expected_volatility": "Low",
        "who_its_for": "Investors nearing a financial goal, or anyone who would be tempted to sell during a downturn.",
        "typical_drawdown": "Historically, portfolios with this mix have seen peak-to-trough declines in the 10-15% range during major downturns, versus 30%+ for all-equity portfolios.",
        "time_horizon_fit": "Best suited to a horizon under 5 years, or for money you can't afford to see drop sharply in the short term.",
        "rebalancing_note": "Because the mix is bond-heavy, drift tends to be slower — checking in quarterly is usually enough.",
    },
    RiskProfile.NEUTRAL: {
        "summary": "Balances growth and stability.",
        "philosophy": "Roughly 60/40 growth-to-defensive split, diversified across regions plus gold and REIT sleeves to smooth out equity/bond correlation risk.",
        "expected_volatility": "Moderate",
        "who_its_for": "Investors with a multi-year horizon who want growth but aren't chasing maximum returns.",
        "typical_drawdown": "Historically, this mix has seen peak-to-trough declines in the 15-25% range during major downturns.",
        "time_horizon_fit": "Fits well for a 5-15 year horizon — long enough to ride out a downturn, short enough that capital preservation still matters.",
        "rebalancing_note": "A meaningful equity sleeve means this mix drifts faster than a conservative one — a semi-annual check is reasonable.",
    },
    RiskProfile.GROWTH: {
        "summary": "Prioritizes long-term capital growth over stability.",
        "philosophy": "Heavily weighted toward equities and real assets (REITs, gold) for higher long-run growth potential. Fixed income is minimal, used mainly to reduce extreme swings.",
        "expected_volatility": "High",
        "who_its_for": "Younger investors or anyone with a long time horizon and high tolerance for drawdowns.",
        "typical_drawdown": "Historically, all-equity-leaning portfolios like this have seen peak-to-trough declines of 30%+ in severe downturns (e.g. 2008, early 2020).",
        "time_horizon_fit": "Best suited to a 15+ year horizon, where there's time to recover from a deep drawdown without needing to sell at a loss.",
        "rebalancing_note": "Higher equity weight means faster drift — worth checking quarterly, especially after a strong equity rally.",
    },
}

EDUCATION_CONTENT = {
    "cash": {"what": "Cash and cash-equivalents (like money market funds) are the most stable, liquid holdings in a portfolio — the closest thing to zero price risk.",
             "why": "Cash cushions the portfolio during downturns and provides dry powder to rebalance into other assets when they get cheap. It typically earns the least over time, so portfolios only hold a small amount."},
    "bonds": {"what": "Fixed income (bonds) are loans to governments or corporations that pay a set interest rate over a defined term.",
              "why": "Bonds are less volatile than stocks and often move differently than equities, especially during stock market downturns — they're the main shock absorber in a balanced portfolio."},
    "cdn_eq": {"what": "Canadian equities are shares of Canadian companies, often concentrated in financials, energy, and materials.",
               "why": "Gives home-market exposure and dividend income, though Canada's market is more concentrated in a few sectors than global markets."},
    "us_eq": {"what": "U.S. equities are shares of American companies, spanning the world's largest and most liquid stock market.",
              "why": "The U.S. market offers the broadest sector diversification (especially technology and healthcare) and has historically been a strong long-term growth driver."},
    "intl_eq": {"what": "International (developed market) equities are shares of companies outside North America — mainly Europe, Japan, and Australia.",
                "why": "Adds geographic diversification so the portfolio isn't dependent on any single country's economic cycle."},
    "em_eq": {"what": "Emerging market equities are shares of companies in developing economies like China, India, and Brazil.",
              "why": "Offers higher long-run growth potential from faster-growing economies, at the cost of higher volatility and political/currency risk."},
    "gold": {"what": "Gold and broader commodities are physical/real assets rather than claims on a company's earnings.",
             "why": "Gold tends to hold value during inflation or crisis periods when stocks and bonds can both struggle, making it a useful diversifier."},
    "reit": {"what": "REITs (Real Estate Investment Trusts) are companies that own and operate income-producing real estate, traded like stocks.",
             "why": "Gives real estate exposure and steady income without directly owning property, and often behaves differently than the broader stock market."},
}

STRESS_SCENARIOS = {
    "2008 Global Financial Crisis": ("2008-09-01", "2009-03-09"),
    "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate-Hike Selloff": ("2022-01-03", "2022-10-12"),
    "2015-16 Oil Crash": ("2015-06-01", "2016-02-11"),
}

RISK_QUESTIONS = [
    {"q": "If your portfolio dropped 20% in a month, you would:",
     "options": [("Sell to avoid further losses", -1), ("Hold and wait it out", 0), ("Buy more while prices are low", 1)]},
    {"q": "How long until you plan to start withdrawing this money?",
     "options": [("Under 3 years", -1), ("3–10 years", 0), ("10+ years", 1)]},
    {"q": "How much investing experience do you have?",
     "options": [("Little to none", -1), ("Some", 0), ("Extensive", 1)]},
    {"q": "What's more important to you?",
     "options": [("Protecting what I have", -1), ("A balance of growth and safety", 0), ("Maximizing long-term growth", 1)]},
    {"q": "How would you feel checking your portfolio during a market downturn?",
     "options": [("Very anxious", -1), ("A bit uneasy but okay", 0), ("Unbothered — it's long-term money", 1)]},
]


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


MAX_SAVED_PORTFOLIOS = 10


def save_portfolio(username, name, amount, risk_profile, horizon, weights, stop_loss=None, custom_tickers=None):
    portfolios = load_saved_portfolios(username)
    if len(portfolios) >= MAX_SAVED_PORTFOLIOS:
        return False
    portfolios.append({
        "name": name, "date_saved": str(date.today()), "amount": amount,
        "risk_profile": risk_profile, "horizon": horizon, "weights": weights, "stop_loss": stop_loss,
        "custom_tickers": custom_tickers or {},
    })
    with open(user_portfolio_file(username), "w") as f:
        json.dump(portfolios, f, indent=2)
    return True


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
    """
    Steps every 2 years instead of 4 broad buckets — 0-2yr is the most
    defensive tilt (-1.0), 20+yr is the most aggressive tilt (+1.0),
    moving in 10 even steps of 2 years each.
    """
    bucket = min(int(years) // 2, 10)
    return -1.0 + (bucket / 10) * 2.0


def horizon_bucket_label(years: float) -> str:
    bucket = min(int(years) // 2, 10)
    lo = bucket * 2
    if bucket == 10:
        return "20+ years"
    return f"{lo}-{lo+2} years"


def expected_return(weights: dict, assumptions: dict = None) -> float:
    """Blended illustrative expected return for the given weights — shifts as
    weights shift with horizon/tilt, or as the user adjusts assumptions on the
    Calculator page. Explicitly NOT a forecast or guarantee."""
    assumptions = assumptions or EXPECTED_RETURNS
    return sum(assumptions.get(k, 0.0) * w for k, w in weights.items())


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


@st.cache_data(ttl=60 * 60 * 24)
def historical_market_condition_score(as_of_date_str: str):
    """Same VIX/momentum scoring logic as market_condition_score, but anchored
    to a specific historical date instead of 'today' — used for quarterly views."""
    try:
        as_of = pd.Timestamp(as_of_date_str)
        start = (as_of - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        end = (as_of + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        vix_hist = yf.Ticker("^VIX").history(start=start, end=end)
        spx_hist = yf.Ticker("^GSPC").history(start=start, end=end)

        # yfinance returns a timezone-aware index; as_of is naive. Comparing
        # the two directly raises, which was silently caught below and made
        # this function always return None — normalize both to naive first.
        if getattr(vix_hist.index, "tz", None) is not None:
            vix_hist.index = vix_hist.index.tz_localize(None)
        if getattr(spx_hist.index, "tz", None) is not None:
            spx_hist.index = spx_hist.index.tz_localize(None)

        vix_hist = vix_hist[vix_hist.index <= as_of]
        spx_hist = spx_hist[spx_hist.index <= as_of]
        if len(vix_hist) < 2 or len(spx_hist) < 2:
            return None

        vix_level = vix_hist["Open"].iloc[-1]
        spx_open = spx_hist["Open"].iloc[-1]
        spx_prev = spx_hist["Open"].iloc[-2]
        momentum = (spx_open / spx_prev) - 1

        vix_score = max(-1.0, min(1.0, (20 - vix_level) / 10))
        momentum_score = max(-1.0, min(1.0, momentum / 0.05))
        return round(0.5 * vix_score + 0.5 * momentum_score, 2)
    except Exception:
        return None


def apply_tactical_tilt(base: dict, horizon_years, market_score):
    """
    Applies the same horizon + market-condition tilt logic to ANY base weights
    dict — the 8 core asset classes are tilted between equity-like/defensive;
    any other keys (e.g. custom tickers) are treated as fixed/untouched and
    the whole thing is renormalized to sum to 1 at the end.
    """
    base = dict(base)
    h_tilt = horizon_tilt(horizon_years)
    m_score = market_score if market_score is not None else 0.0

    composite_tilt = 0.65 * h_tilt + 0.35 * m_score
    composite_tilt = max(-1.0, min(1.0, composite_tilt))

    equity_like_keys = [k for k in ["cdn_eq", "us_eq", "intl_eq", "em_eq", "reit"] if k in base]
    defensive_keys = [k for k in ["bonds", "cash", "gold"] if k in base]

    shift = composite_tilt * TACTICAL_RANGE
    total_equity_base = sum(base[k] for k in equity_like_keys)
    total_defensive_base = sum(base[k] for k in defensive_keys)
    shift = max(min(shift, total_defensive_base * 0.9), -total_equity_base * 0.9) if (total_equity_base > 0 and total_defensive_base > 0) else 0.0

    adjusted = base.copy()
    if total_equity_base > 0:
        for k in equity_like_keys:
            adjusted[k] = base[k] + shift * (base[k] / total_equity_base)
    if total_defensive_base > 0:
        for k in defensive_keys:
            adjusted[k] = base[k] - shift * (base[k] / total_defensive_base)

    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()} if total > 0 else adjusted


def build_portfolio(risk_profile, horizon_years, market_score):
    return apply_tactical_tilt(STRATEGIC_MIX[risk_profile], horizon_years, market_score)


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
def fetch_price_history(tickers_tuple, period=None, start=None, end=None):
    tickers = [t for t in tickers_tuple if t is not None]
    if start is not None:
        data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    else:
        data = yf.download(tickers, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


def run_backtest(weights: dict, tickers_map: dict, years: int, stop_loss_pct=None):
    period = "1y" if years <= 1 else ("5y" if years <= 5 else "10y")
    tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}

    try:
        prices = fetch_price_history(tuple(tickers_used.values()), period=period)
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
        bench_prices = fetch_price_history(tuple(bench_tickers), period=period).dropna()
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
# 6. STRESS TESTING
# =======================================================================
def run_stress_test(weights: dict, tickers_map: dict, amount: float, start: str, end: str):
    tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None and weights[k] > 0}
    per_asset_impact = {}
    missing = []
    weighted_impact = 0.0
    covered_weight = 0.0

    for k, ticker in tickers_used.items():
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end)
            if len(hist) < 2:
                missing.append(ASSET_LABELS.get(k, k))
                continue
            pct_change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
            per_asset_impact[k] = pct_change
            weighted_impact += weights[k] * pct_change
            covered_weight += weights[k]
        except Exception:
            missing.append(ASSET_LABELS.get(k, k))

    cash_weight = weights.get("cash", 0.0)
    covered_weight += cash_weight  # cash assumed flat (0% change) during the scenario

    if covered_weight > 0:
        portfolio_pct_impact = weighted_impact / covered_weight
    else:
        portfolio_pct_impact = None

    return {
        "per_asset_impact": per_asset_impact, "portfolio_pct_impact": portfolio_pct_impact,
        "dollar_impact": amount * portfolio_pct_impact if portfolio_pct_impact is not None else None,
        "missing": missing, "covered_weight": covered_weight,
    }


# =======================================================================
# 7. SENSITIVITY INDEX (rate beta + market beta per asset class)
# =======================================================================
@st.cache_data(ttl=60 * 60 * 12)
def compute_sensitivity(tickers_tuple, period="3y"):
    tickers = [t for t in tickers_tuple if t is not None]
    if not tickers:
        return {}

    def _clean(series):
        # Normalize timezone so US-listed and Canadian-listed tickers align on date,
        # and drop only that series' own missing values (not shared with others).
        s = series.copy()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        return s.dropna()

    try:
        prices = fetch_price_history(tuple(tickers), period=period)
    except Exception:
        return {}

    tnx_changes = spx_returns = oil_returns = None
    try:
        tnx = yf.Ticker("^TNX").history(period=period)["Close"]
        tnx_changes = _clean(tnx.diff())
    except Exception:
        pass
    try:
        spx = yf.Ticker("^GSPC").history(period=period)["Close"]
        spx_returns = _clean(spx.pct_change())
    except Exception:
        pass
    try:
        oil = yf.Ticker("CL=F").history(period=period)["Close"]
        oil_returns = _clean(oil.pct_change())
    except Exception:
        pass

    results = {}
    for t in tickers:
        if t not in prices.columns:
            results[t] = {"rate_beta": None, "market_beta": None, "oil_beta": None}
            continue

        asset_ret = _clean(prices[t].pct_change())

        rate_beta = market_beta = oil_beta = None
        if tnx_changes is not None:
            aligned = pd.concat([asset_ret, tnx_changes], axis=1, join="inner").dropna()
            if len(aligned) > 20:
                rate_beta = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)[0]
        if spx_returns is not None:
            aligned = pd.concat([asset_ret, spx_returns], axis=1, join="inner").dropna()
            if len(aligned) > 20:
                market_beta = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)[0]
        if oil_returns is not None:
            aligned = pd.concat([asset_ret, oil_returns], axis=1, join="inner").dropna()
            if len(aligned) > 20:
                oil_beta = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)[0]

        results[t] = {"rate_beta": rate_beta, "market_beta": market_beta, "oil_beta": oil_beta}

    return results


# =======================================================================
# 8. REBALANCING SIMULATOR
# =======================================================================
def simulate_rebalancing(weights: dict, tickers_map: dict, amount: float, date_saved: str):
    tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}
    try:
        # yfinance's `end` is exclusive, so a portfolio saved today/yesterday can
        # return zero rows unless we push the end date forward by a day.
        end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        prices = fetch_price_history(tuple(tickers_used.values()), start=date_saved, end=end_date)
        prices = prices.dropna()
        if len(prices) < 2:
            days_elapsed = (date.today() - pd.Timestamp(date_saved).date()).days
            return {"error": f"Only {days_elapsed} day(s) have passed since this portfolio was saved — "
                              f"not enough time for a full trading day to have closed yet. Try again tomorrow, "
                              f"or pick a portfolio saved further in the past."}

        start_prices = prices.iloc[0]
        current_prices = prices.iloc[-1]

        shares = {}
        for k, ticker in tickers_used.items():
            dollar_amount = amount * weights[k]
            shares[k] = dollar_amount / start_prices[ticker]

        current_values = {}
        for k, ticker in tickers_used.items():
            current_values[k] = shares[k] * current_prices[ticker]
        cash_value = amount * weights.get("cash", 0.0)  # cash assumed flat
        current_values["cash"] = cash_value

        total_current = sum(current_values.values())
        current_weights = {k: v / total_current for k, v in current_values.items()}

        drift = {k: current_weights.get(k, 0.0) - weights.get(k, 0.0) for k in weights}
        trades_needed = {k: (weights[k] - current_weights.get(k, 0.0)) * total_current for k in weights}

        return {
            "current_values": current_values, "current_weights": current_weights,
            "total_current": total_current, "drift": drift, "trades_needed": trades_needed,
            "start_date": prices.index[0].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"error": str(e)}


# =======================================================================
# 9. CENTRAL BANK POLICY RATES (Fed, BoC, BoE via FRED)
# =======================================================================
@st.cache_data(ttl=60 * 60 * 12)
def fetch_boc_rate():
    """
    Bank of Canada's own FRED mirror series (IRSTCB01CAM156N) was discontinued
    after Dec 2023, so this pulls directly from the Bank of Canada's live
    Valet API instead — series CBC20210 is the target for the overnight rate,
    updated the day after each of the Bank's 8 scheduled announcements per year.
    """
    import requests
    try:
        resp = requests.get(
            "https://www.bankofcanada.ca/valet/observations/CBC20210/json",
            params={"recent": 1}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data["observations"][-1]
        rate = float(obs["CBC20210"]["v"])
        as_of = obs["d"]
        return rate, as_of
    except Exception:
        return None


@st.cache_data(ttl=60 * 60 * 12)
def fetch_central_bank_rates():
    """Pulls the latest policy rate for all G10 central banks. Bank of Canada
    comes from its own live Valet API (see fetch_boc_rate); the rest come from
    FRED. Returns a dict of {bank: (rate, as_of_date)} — any bank that fails
    to fetch is simply omitted rather than breaking the page."""
    import pandas_datareader.data as web
    results = {}

    boc = fetch_boc_rate()
    if boc is not None:
        results["Bank of Canada (Overnight Rate Target)"] = boc

    series = {
        "Federal Reserve (Fed Funds Rate)": "DFF",
        "Bank of England (Bank Rate)": "IRSTCB01GBM156N",
        "European Central Bank (Deposit Rate)": "ECBDFR",
        "Bank of Japan (Policy Rate)": "IRSTCB01JPM156N",
        "Reserve Bank of Australia (Cash Rate)": "IRSTCB01AUM156N",
        "Reserve Bank of New Zealand (OCR)": "IRSTCB01NZM156N",
        "Swiss National Bank (Policy Rate)": "IRSTCB01CHM156N",
        "Norges Bank (Key Policy Rate)": "IRSTCB01NOM156N",
        "Riksbank (Policy Rate)": "IRSTCB01SEM156N",
    }
    end = date.today()
    start = end - timedelta(days=800)  # wider window — several of these update monthly/quarterly with a lag
    for name, code in series.items():
        try:
            data = web.DataReader(code, "fred", start, end).dropna()
            if not data.empty:
                latest = data.iloc[-1]
                results[name] = (float(latest.iloc[0]), data.index[-1].strftime("%Y-%m-%d"))
        except Exception:
            continue
    return results


@st.cache_data(ttl=60 * 15)  # 15-minute refresh so this stays close to live through the day
def fetch_g10_currency_rates():
    """G10 currency pairs, all quoted per 1 USD (except EUR/GBP which are
    quoted as USD per unit, matching standard FX convention)."""
    pairs = {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
        "USD/CAD": "USDCAD=X", "AUD/USD": "AUDUSD=X", "NZD/USD": "NZDUSD=X",
        "USD/CHF": "USDCHF=X", "USD/SEK": "USDSEK=X", "USD/NOK": "USDNOK=X",
    }
    results = {}
    for label, ticker in pairs.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                latest_price = hist["Close"].iloc[-1]
                prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else latest_price
                pct_change = (latest_price / prev_price - 1) * 100
                results[label] = (latest_price, pct_change)
        except Exception:
            continue
    return results


# =======================================================================
# 10. NEWS
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


DAILY_BRIEFING_FEEDS = {
    "Investment Product Advisory": "https://www.investing.com/rss/320.rss",       # ETF Analysis & Opinion
    "FX Market": "https://www.investing.com/rss/news_1.rss",                     # Forex News
    "Metals & Mining": "https://www.investing.com/rss/commodities_Metals.rss",   # Metals Analysis
    "Bond Market": "https://www.investing.com/rss/bonds.rss",                   # Bonds Analysis & Opinion
}


@st.cache_data(ttl=60 * 60 * 4)
def fetch_daily_briefing():
    """
    Pulls real, topic-specific RSS feeds (not ticker-based proxies) so each
    section actually matches its subject — e.g. Investment Product Advisory
    pulls genuine ETF/fund-launch/AUM news, not just an asset manager's stock
    price headlines.
    """
    import requests
    import xml.etree.ElementTree as ET

    briefing = {}
    for sector, feed_url in DAILY_BRIEFING_FEEDS.items():
        headlines = []
        try:
            resp = requests.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:6]:
                title = item.findtext("title", default="Untitled")
                link = item.findtext("link", default="")
                author = item.findtext("author", default="")
                pub_date = item.findtext("pubDate", default="")
                headlines.append({"title": title, "link": link, "publisher": author, "pub_date": pub_date})
        except Exception:
            pass
        briefing[sector] = headlines
    return briefing


# =======================================================================
# 11. EXPORT (CSV + PDF)
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
    pdf.cell(0, 10, "Wealthscope - Portfolio Allocation Report", ln=True)
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
# 12. PAGE CONFIG + STYLE
# =======================================================================
st.set_page_config(page_title="Wealthscope", page_icon="📊", layout="wide")

LOGO_PATH = "assets/logo.png"

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEME_PALETTES = {
    "dark": {
        "bg": "#0a0e1a", "bg_gradient_end": "#0e1117",
        "card": "#171b24", "card_gradient_end": "#131722",
        "border": "#2b2f38", "border_hover": "#3b82f6",
        "text": "#e8eaed", "text_muted": "#9aa0ab", "text_faint": "#6b7280",
        "accent": "#60a5fa", "accent2": "#3b82f6", "accent_strong": "#2563eb", "accent_deep": "#1d4ed8",
        "sidebar_start": "#10131a", "sidebar_end": "#0c0e13",
        "input_bg": "#171b24", "shadow": "rgba(59, 130, 246, 0.18)",
    },
    "light": {
        "bg": "#ffffff", "bg_gradient_end": "#eef2fb",
        "card": "#ffffff", "card_gradient_end": "#f4f7fd",
        "border": "#dbe3f0", "border_hover": "#2554c7",
        "text": "#132049", "text_muted": "#4b5a80", "text_faint": "#7688ab",
        "accent": "#2554c7", "accent2": "#1e40af", "accent_strong": "#1a3a99", "accent_deep": "#132c7a",
        "sidebar_start": "#f4f7fd", "sidebar_end": "#e9eefb",
        "input_bg": "#ffffff", "shadow": "rgba(37, 84, 199, 0.15)",
    },
}


def render_theme_css(theme_name: str):
    p = THEME_PALETTES[theme_name]
    st.markdown(f"""
        <style>
            .main, [data-testid="stAppViewContainer"] {{
                background: linear-gradient(160deg, {p['bg']} 0%, {p['bg_gradient_end']} 100%);
            }}
            .main *:not(h1):not(h2):not(h3), [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
            [data-testid="stAppViewContainer"] label, [data-testid="stMarkdownContainer"]:not(:has(h1)):not(:has(h2)):not(:has(h3)) {{
                color: {p['text']};
            }}
            h1, h2, h3 {{ letter-spacing: -0.3px; }}
            h2, h3 {{ color: {p['accent']}; }}
            [data-testid="stHeaderActionElements"] {{ display: none; }}

            h1 {{
                background: linear-gradient(90deg, {p['accent']} 0%, {p['accent_strong']} 50%, {p['accent']} 100%);
                background-size: 200% auto;
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: shine 6s linear infinite;
            }}
            @keyframes shine {{ to {{ background-position: 200% center; }} }}

            [data-testid="stMetricValue"] {{ color: {p['text']}; }}
            [data-testid="stMetricLabel"] {{ color: {p['text_muted']}; }}
            [data-testid="stMetric"] {{
                background: linear-gradient(160deg, {p['card']} 0%, {p['card_gradient_end']} 100%);
                border: 1px solid {p['border']};
                border-radius: 10px; padding: 12px 16px;
                transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            [data-testid="stMetric"]:hover {{
                transform: translateY(-2px);
                border-color: {p['border_hover']};
                box-shadow: 0 4px 16px {p['shadow']};
            }}

            .stButton>button {{
                background: linear-gradient(135deg, {p['accent_strong']}, {p['accent']});
                color: white; border: none; border-radius: 8px;
                font-weight: 600; padding: 0.5em 1.5em;
                transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease;
            }}
            .stButton>button:hover {{
                background: linear-gradient(135deg, {p['accent_deep']}, {p['accent_strong']});
                color: white; transform: translateY(-1px);
                box-shadow: 0 4px 14px {p['shadow']};
            }}
            .stButton>button:active {{ transform: translateY(0px); }}

            .stDownloadButton>button {{
                background-color: {p['card']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: 8px;
                transition: border-color 0.15s ease;
            }}
            .stDownloadButton>button:hover {{ border-color: {p['accent']}; }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {p['sidebar_start']} 0%, {p['sidebar_end']} 100%);
                border-right: 1px solid {p['border']};
            }}
            section[data-testid="stSidebar"] * {{ color: {p['text']}; }}

            [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
            .stSelectbox [data-baseweb="select"] {{
                background-color: {p['input_bg']} !important;
                color: {p['text']} !important;
                border-color: {p['border']} !important;
            }}

            /* BaseWeb renders the OPEN dropdown list in a portal outside .stSelectbox,
               so it needs its own unscoped rule to pick up the current theme. */
            [data-baseweb="popover"] [data-baseweb="menu"] {{
                background-color: {p['card']} !important;
                border: 1px solid {p['border']} !important;
            }}
            [data-baseweb="popover"] [data-baseweb="menu"] li {{
                background-color: {p['card']} !important;
                color: {p['text']} !important;
            }}
            [data-baseweb="popover"] [data-baseweb="menu"] li:hover {{
                background-color: {p['bg_gradient_end']} !important;
            }}

            .edu-card, .kpi-card {{
                background: linear-gradient(160deg, {p['card']} 0%, {p['card_gradient_end']} 100%);
                border: 1px solid {p['border']}; border-radius: 10px;
                padding: 16px 20px; margin-bottom: 14px;
                transition: transform 0.15s ease, border-color 0.15s ease;
            }}
            .edu-card:hover {{ transform: translateY(-2px); border-color: {p['border_hover']}; }}
            .edu-card h4 {{ margin: 0 0 8px 0; color: {p['accent']}; }}

            .kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
            .kpi-box {{
                background: linear-gradient(160deg, {p['card']} 0%, {p['card_gradient_end']} 100%);
                border: 1px solid {p['border']}; border-radius: 10px;
                padding: 14px 18px; flex: 1; min-width: 180px;
                transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            .kpi-box:hover {{
                transform: translateY(-2px);
                border-color: {p['border_hover']};
                box-shadow: 0 4px 16px {p['shadow']};
            }}
            .kpi-box .label {{ color: {p['text_muted']}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .kpi-box .value {{ color: {p['text']}; font-size: 24px; font-weight: 700; margin-top: 4px; }}
            .kpi-box .asof {{ color: {p['text_faint']}; font-size: 11px; margin-top: 2px; }}

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.login-tagline) {{
                background: linear-gradient(160deg, {p['card']} 0%, {p['card_gradient_end']} 100%);
                border: 1px solid {p['border']} !important; border-radius: 16px !important;
                padding: 24px 20px; margin-top: 20px;
                box-shadow: 0 8px 32px {p['shadow']};
            }}
            .login-tagline {{ color: {p['text_muted']}; font-size: 14px; letter-spacing: 1px; text-transform: uppercase; }}

            [data-testid="stAppViewContainer"] {{ animation: fadeIn 0.35s ease-in; }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        </style>
    """, unsafe_allow_html=True)


render_theme_css(st.session_state.theme)

config = load_or_create_user_config()
authenticator = stauth.Authenticate(
    config["credentials"], config["cookie"]["name"], config["cookie"]["key"], config["cookie"]["expiry_days"]
)

if not st.session_state.get("authentication_status"):
    left_pad, center, right_pad = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            if os.path.exists(LOGO_PATH):
                import base64
                with open(LOGO_PATH, "rb") as f:
                    logo_b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<div style="display:flex; justify-content:center; align-items:center; width:100%;">'
                    f'<img src="data:image/png;base64,{logo_b64}" width="140">'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<h1 style='text-align:center; margin-bottom:0;'>Wealthscope</h1>", unsafe_allow_html=True)
            st.markdown("<p class='login-tagline' style='text-align:center;'>Insight. Growth. Wealth.</p>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center; font-size:15px; margin-top:8px;'>"
                "Build, backtest, and stress-test investment portfolios with real market data — all in one place."
                "</p>", unsafe_allow_html=True
            )
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
# LOGGED IN FROM HERE ON
# =======================================================================
username = st.session_state["username"]

for key, default in [
    ("current_weights", None), ("current_amount", 20000.0), ("current_risk", RiskProfile.NEUTRAL.value),
    ("current_horizon", 10), ("custom_tickers", {}), ("stop_loss_pct", 15.0), ("is_customized", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "custom_expected_returns" not in st.session_state:
    st.session_state.custom_expected_returns = dict(EXPECTED_RETURNS)

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)
    else:
        st.markdown("### Wealthscope")
    st.write(f"Logged in as **{st.session_state['name']}**")

    authenticator.logout()
    st.divider()

    _p = THEME_PALETTES[st.session_state.theme]
    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Risk Questionnaire", "Calculator", "Backtest & Risk", "Stress Testing",
                 "Efficient Frontier", "Sensitivity Index", "Rebalancing Simulator", "Quarterly Views",
                 "Compare Profiles", "Market News", "Daily Briefing", "Saved Portfolios", "Portfolio Education", "Strategy Guide"],
        icons=["speedometer2", "clipboard-check", "calculator", "graph-up-arrow", "exclamation-triangle",
               "bullseye", "activity", "arrow-repeat", "calendar3",
               "bar-chart-steps", "newspaper", "sun", "save", "mortarboard", "book"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": _p["sidebar_start"]},
            "icon": {"color": _p["accent"], "font-size": "15px"},
            "nav-link": {"font-size": "13px", "color": _p["text"], "--hover-color": _p["card"]},
            "nav-link-selected": {"background-color": _p["accent_strong"]},
        },
    )

st.caption("Builds a target asset allocation from your risk profile, horizon, and the prior trading day's market conditions.")


def select_portfolio_for_page(widget_key: str):
    """
    Renders a dropdown letting the user choose between the portfolio currently
    active on the Calculator page and any of their saved portfolios. Returns
    a dict with weights/tickers/labels/amount/risk/horizon for whichever is
    chosen, or None if nothing is available yet.
    """
    saved = load_saved_portfolios(username)
    options = []
    if st.session_state.current_weights:
        options.append("Current Calculator Portfolio")
    options += [p["name"] for p in saved]

    if not options:
        st.info("No portfolio available yet — build one on the Calculator page or save one first.")
        return None

    choice = st.selectbox("Portfolio to use", options, key=f"portfolio_choice_{widget_key}")

    if choice == "Current Calculator Portfolio":
        return {
            "weights": st.session_state.current_weights,
            "tickers_map": st.session_state.get("_combined_tickers", TICKERS),
            "labels_map": st.session_state.get("_combined_labels", ASSET_LABELS),
            "amount": st.session_state.current_amount,
            "risk_profile": st.session_state.current_risk,
            "horizon": st.session_state.current_horizon,
            "stop_loss": st.session_state.stop_loss_pct,
            "source": "calculator",
        }
    else:
        p = next(sp for sp in saved if sp["name"] == choice)
        tickers_map = dict(TICKERS)
        labels_map = dict(ASSET_LABELS)
        for key, v in p.get("custom_tickers", {}).items():
            tickers_map[key] = v["ticker"]
            labels_map[key] = f"{v['name']} ({v['ticker']})"
        return {
            "weights": p["weights"], "tickers_map": tickers_map, "labels_map": labels_map,
            "amount": p["amount"], "risk_profile": p["risk_profile"], "horizon": p["horizon"],
            "stop_loss": p.get("stop_loss"), "source": "saved", "date_saved": p["date_saved"],
        }


# =======================================================================
# PAGE: DASHBOARD
# =======================================================================
if page == "Dashboard":
    st.subheader(f"Welcome back, {st.session_state['name']}")

    score, as_of, vix_level = market_condition_score(str(date.today()))
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Market Condition Score", f"{score:+.2f}" if score is not None else "N/A")
    with m2:
        st.metric("As-of Date (9:00 AM open)", as_of if score is not None else "unavailable")
    with m3:
        st.metric("Prior-Day VIX (open)", f"{vix_level:.2f}" if vix_level else "N/A")

    st.markdown("**Central Bank Policy Rates**")
    rates = fetch_central_bank_rates()
    if rates:
        rate_items = list(rates.items())
        for row_start in range(0, len(rate_items), 5):
            row_items = rate_items[row_start:row_start + 5]
            cols = st.columns(len(row_items))
            for col, (name, (rate, as_of_rate)) in zip(cols, row_items):
                with col:
                    st.markdown(f"""
                        <div class="kpi-box">
                            <div class="label">{name}</div>
                            <div class="value">{rate:.2f}%</div>
                            <div class="asof">as of {as_of_rate}</div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.caption("Central bank rate data unavailable right now.")

    st.divider()

    st.markdown("**G10 Currency Rates**")
    st.caption("Refreshes every 15 minutes throughout the trading day.")
    fx = fetch_g10_currency_rates()
    if fx:
        fx_items = list(fx.items())
        for row_start in range(0, len(fx_items), 5):
            row_items = fx_items[row_start:row_start + 5]
            cols = st.columns(len(row_items))
            for col, (pair, (price, pct_change)) in zip(cols, row_items):
                with col:
                    arrow = "▲" if pct_change >= 0 else "▼"
                    color = "#4ade80" if pct_change >= 0 else "#f87171"
                    st.markdown(f"""
                        <div class="kpi-box">
                            <div class="label">{pair}</div>
                            <div class="value">{price:.4f}</div>
                            <div class="asof" style="color:{color}">{arrow} {abs(pct_change):.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.caption("Currency rate data unavailable right now.")

    st.divider()

    portfolios = load_saved_portfolios(username)
    if not portfolios:
        st.info("You don't have any saved portfolios yet. Head to the Calculator to build one, or try the Risk Questionnaire to get a starting recommendation.")
    else:
        names = [p["name"] for p in portfolios]
        default_idx = len(names) - 1  # most recent by default
        chosen_name = st.selectbox("View saved portfolio", names, index=default_idx)
        chosen = next(p for p in portfolios if p["name"] == chosen_name)

        st.caption(f"Saved {chosen['date_saved']} — {chosen['risk_profile']} profile, {chosen['horizon']}yr horizon, ${chosen['amount']:,.0f}")

        left, right = st.columns([1, 1.3])
        w = chosen["weights"]
        dash_labels = dict(ASSET_LABELS)
        for key, v in chosen.get("custom_tickers", {}).items():
            dash_labels[key] = f"{v['name']} ({v['ticker']})"
        with left:
            fig = go.Figure(data=[go.Pie(labels=[dash_labels.get(k, k) for k in w], values=[v * 100 for v in w.values()],
                                          hole=0.45, textinfo="label+percent")])
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]))
            st.plotly_chart(fig, use_container_width=True, key="dash_pie")
        with right:
            exp_ret = expected_return({k: v for k, v in w.items() if k in EXPECTED_RETURNS})
            st.metric("Illustrative Expected Annual Return", f"{exp_ret*100:.2f}%",
                       help="Based on long-run capital market assumptions per asset class — not a forecast.")
            df = pd.DataFrame({"Asset": [dash_labels.get(k, k) for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()]})
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(f"You have {len(portfolios)} saved portfolio{'s' if len(portfolios) != 1 else ''} total — see the Saved Portfolios page for all of them.")


# =======================================================================
# PAGE: RISK QUESTIONNAIRE
# =======================================================================
elif page == "Risk Questionnaire":
    st.subheader("Risk Questionnaire")
    st.caption("Answer a few quick questions to get a suggested risk profile — you can always override it manually on the Calculator page.")

    answers = []
    for i, item in enumerate(RISK_QUESTIONS):
        choice = st.radio(item["q"], [opt[0] for opt in item["options"]], key=f"rq_{i}")
        score_val = dict(item["options"])[choice]
        answers.append(score_val)

    if st.button("Get My Risk Profile"):
        total_score = sum(answers)
        if total_score <= -2:
            suggested = RiskProfile.CONSERVATIVE
        elif total_score >= 2:
            suggested = RiskProfile.GROWTH
        else:
            suggested = RiskProfile.NEUTRAL

        st.session_state["_suggested_profile"] = suggested.value
        st.success(f"Based on your answers, your suggested risk profile is **{suggested.value}**.")

    if "_suggested_profile" in st.session_state:
        notes = STRATEGY_NOTES[RiskProfile(st.session_state["_suggested_profile"])]
        st.markdown(f"**{st.session_state['_suggested_profile']}** — {notes['summary']}")
        st.markdown(notes["philosophy"])
        if st.button("Use This Profile on the Calculator"):
            st.session_state.current_risk = st.session_state["_suggested_profile"]
            st.success("Set as your active risk profile — head to the Calculator page to build the portfolio.")


# =======================================================================
# PAGE: CALCULATOR
# =======================================================================
elif page == "Calculator":
    # Must run before the exp_ret_* number_input widgets below are instantiated —
    # Streamlit forbids overwriting a widget's session_state value in the same
    # run after that widget has already been created, so the reset has to happen
    # here, one run ahead of the button click that requests it.
    if st.session_state.get("_reset_assumptions"):
        st.session_state.custom_expected_returns = dict(EXPECTED_RETURNS)
        for k in EXPECTED_RETURNS:
            st.session_state[f"exp_ret_{k}"] = EXPECTED_RETURNS[k] * 100
        st.session_state["_reset_assumptions"] = False

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Investment Amount ($)", min_value=1.0, max_value=10_000_000.0, value=min(st.session_state.current_amount, 10_000_000.0), step=500.0)
    with col2:
        risk_choice = st.selectbox("Risk Profile", [p.value for p in RiskProfile],
                                    index=[p.value for p in RiskProfile].index(st.session_state.current_risk))
    with col3:
        horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=50,
                                   value=min(st.session_state.current_horizon, 50), step=1)
    st.caption(f"Horizon bucket: **{horizon_bucket_label(horizon)}** — the allocation tilt shifts every 2 years of horizon, not just at broad thresholds.")

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
        st.subheader("Add a Custom Ticker")
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

        combined_tickers = dict(TICKERS)
        combined_labels = dict(ASSET_LABELS)
        for key, v in st.session_state.custom_tickers.items():
            combined_tickers[key] = v["ticker"]
            combined_labels[key] = f"{v['name']} ({v['ticker']})"

        weights = {k: v for k, v in weights.items() if k in combined_labels}

        st.divider()
        heading = "Your Custom Allocation" if st.session_state.is_customized else "Recommended Allocation"
        st.subheader(heading)

        exp_ret = expected_return({k: v for k, v in weights.items() if k in st.session_state.custom_expected_returns},
                                   st.session_state.custom_expected_returns)
        st.metric("Illustrative Expected Annual Return", f"{exp_ret*100:.2f}%",
                   help="Based on long-run capital market assumptions per asset class, weighted by your current allocation. Not a forecast or guarantee — shifts as your horizon bucket, weights, or return assumptions change.")

        with st.expander("Why this number, and how much should you trust it?"):
            st.markdown(
                "This is a **weighted average** of long-run historical return assumptions for each asset class — "
                "not a model prediction, and not specific to your holdings' actual future performance. "
                "The formula is simply:"
            )
            st.latex(r"\text{Expected Return} = \sum_{i} (\text{Weight}_i \times \text{Assumed Return}_i)")
            exp_rows = [
                {"Asset Class": combined_labels.get(k, k), "Weight": f"{weights[k]*100:.1f}%",
                 "Assumed Long-Run Return": f"{st.session_state.custom_expected_returns[k]*100:.1f}%",
                 "Contribution": f"{weights[k]*st.session_state.custom_expected_returns[k]*100:.2f}%"}
                for k in weights if k in st.session_state.custom_expected_returns
            ]
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
            st.markdown(
                "**Why these specific numbers?** They're rough, commonly-cited long-run historical averages per "
                "asset class (e.g. broad equities ~7-9%/yr, bonds ~4%/yr over multi-decade periods) — the same "
                "order of magnitude you'd see in most institutional capital market assumption sheets, simplified "
                "for this project. **You can adjust them below** if you have your own view.\n\n"
                "**Should you trust it as a forecast?** No — treat it as a rough anchor, not a promise. Actual "
                "annual returns vary enormously year to year (a portfolio 'expected' to return 7% might return "
                "-15% or +25% in any given year). For a data-backed look at how this exact portfolio actually "
                "performed historically, use the **Backtest & Risk** page instead, which pulls real price history "
                "rather than assumptions."
            )

        with st.expander("Adjust Return Assumptions"):
            st.caption("Override the assumed long-run annual return for each asset class — the expected return above updates immediately.")
            edit_cols = st.columns(4)
            for i, k in enumerate(EXPECTED_RETURNS):
                with edit_cols[i % 4]:
                    st.session_state.custom_expected_returns[k] = st.number_input(
                        ASSET_LABELS[k], min_value=-20.0, max_value=30.0,
                        value=st.session_state.custom_expected_returns[k] * 100, step=0.1,
                        key=f"exp_ret_{k}", format="%.1f",
                    ) / 100
            if st.button("Reset to Default Assumptions"):
                st.session_state["_reset_assumptions"] = True
                st.rerun()

        left, right = st.columns([1, 1.3])
        with left:
            fig = go.Figure(data=[go.Pie(
                labels=[combined_labels[k] for k in weights], values=[v * 100 for v in weights.values()],
                hole=0.45, textinfo="label+percent",
            )])
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]))
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
        st.subheader("Custom Weights")
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
            st.divider()
            st.subheader("Data Validation")
            for issue in issues:
                st.warning(issue)

        st.divider()
        st.subheader("Stop-Loss Threshold")
        stop_loss_pct = st.number_input("Stop-loss (% decline from peak value)", min_value=0.0, max_value=100.0,
                                         value=st.session_state.stop_loss_pct, step=1.0)
        st.session_state.stop_loss_pct = stop_loss_pct
        st.caption("Used on the Backtest tab to flag whether this threshold would have been breached historically.")

        st.divider()
        st.subheader("Export")
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
        st.subheader("Save This Portfolio")
        existing_count = len(load_saved_portfolios(username))
        st.caption(f"{existing_count}/{MAX_SAVED_PORTFOLIOS} saved portfolios used.")
        save_name = st.text_input("Portfolio name", placeholder="e.g. My Growth Mix")
        if st.button("Save Portfolio"):
            if not save_name.strip():
                st.warning("Give your portfolio a name first.")
            elif existing_count >= MAX_SAVED_PORTFOLIOS:
                st.error(f"You've reached the {MAX_SAVED_PORTFOLIOS}-portfolio limit — delete one from the Saved Portfolios page before saving another.")
            else:
                saved_ok = save_portfolio(username, save_name.strip(), amount, risk_choice, horizon, weights,
                                           stop_loss_pct, st.session_state.custom_tickers)
                if saved_ok:
                    st.success(f"Saved '{save_name}' to your account.")
                else:
                    st.error(f"You've reached the {MAX_SAVED_PORTFOLIOS}-portfolio limit — delete one from the Saved Portfolios page before saving another.")

        st.session_state["_combined_tickers"] = combined_tickers
        st.session_state["_combined_labels"] = combined_labels
        st.session_state["current_weights"] = weights


# =======================================================================
# PAGE: BACKTEST & RISK
# =======================================================================
elif page == "Backtest & Risk":
    st.subheader("Historical Backtest")
    selection = select_portfolio_for_page("backtest")
    if selection:
        weights = selection["weights"]
        tickers_map = selection["tickers_map"]
        stop_loss_pct = selection["stop_loss"] or 15.0
        bt_years = st.radio("Lookback period", [1, 5, 10], index=1, horizontal=True, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")

        if st.button("Run Backtest"):
            with st.spinner("Pulling historical data..."):
                result = run_backtest(weights, tickers_map, bt_years, stop_loss_pct)
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
                                   font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]), legend=dict(orientation="h"), yaxis_title="Growth of $100")
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
                    st.error(f"Your stop-loss threshold of {stop_loss_pct:.0f}% would have been breached on {result['stop_loss_breach_date']} (max drawdown over this period: {result['max_drawdown']*100:.1f}%).")
                else:
                    st.success(f"Your stop-loss threshold of {stop_loss_pct:.0f}% was not breached over this period (max drawdown: {result['max_drawdown']*100:.1f}%).")

                st.caption(f"Sharpe ratio assumes a {RISK_FREE_RATE*100:.1f}% annualized risk-free rate. Past performance is not indicative of future results.")


# =======================================================================
# PAGE: STRESS TESTING
# =======================================================================
elif page == "Stress Testing":
    st.subheader("Stress Testing")
    st.caption("Applies the actual historical returns of your held assets during past crisis periods to your current weights — showing what would have happened, not a prediction of what will.")

    selection = select_portfolio_for_page("stress")
    if selection:
        weights = selection["weights"]
        tickers_map = selection["tickers_map"]
        labels_map = selection["labels_map"]
        amount = selection["amount"]

        scenario = st.selectbox("Scenario", list(STRESS_SCENARIOS.keys()))
        start, end = STRESS_SCENARIOS[scenario]
        st.caption(f"Window: {start} to {end}")

        if st.button("Run Stress Test"):
            with st.spinner("Pulling historical crisis-period data..."):
                result = run_stress_test(weights, tickers_map, amount, start, end)
            st.session_state["_stress_result"] = (scenario, result)

        cached = st.session_state.get("_stress_result")
        if cached:
            cached_scenario, result = cached
            if result["portfolio_pct_impact"] is None:
                st.error("No usable price data for this scenario — the held tickers may not have existed yet.")
            else:
                st.metric(f"Estimated Portfolio Impact ({cached_scenario})",
                          f"{result['portfolio_pct_impact']*100:+.1f}%",
                          delta=f"${result['dollar_impact']:,.0f}")
                st.caption(f"Based on {result['covered_weight']*100:.0f}% of your portfolio's weight (some assets may lack data this far back).")

                if result["missing"]:
                    st.warning(f"No historical data available for: {', '.join(result['missing'])} — excluded from this estimate.")

                asset_df = pd.DataFrame({
                    "Asset Class": [labels_map.get(k, k) for k in result["per_asset_impact"]],
                    "Change During Scenario": [f"{v*100:+.1f}%" for v in result["per_asset_impact"].values()],
                })
                st.dataframe(asset_df, use_container_width=True, hide_index=True)


# =======================================================================
# PAGE: EFFICIENT FRONTIER
# =======================================================================
elif page == "Efficient Frontier":
    st.subheader("Efficient Frontier (Monte Carlo)")

    selection = select_portfolio_for_page("frontier")
    if selection:
        weights = selection["weights"]
        tickers_map = selection["tickers_map"]
        labels_map = selection["labels_map"]
        held_names = [labels_map.get(k, k) for k, w in weights.items() if w > 0]

        fr_years = st.radio("Lookback period", [1, 5, 10], index=1, horizontal=True,
                             format_func=lambda y: f"{y} year{'s' if y > 1 else ''}", key="frontier_years")

        st.caption(
            f"Frontier for: **{selection['risk_profile']} profile, ${selection['amount']:,.0f}, "
            f"{selection['horizon']}yr horizon** — simulated using the currently held assets "
            f"({', '.join(held_names)}) over the selected lookback period. "
            f"3,000 random-weight portfolios are simulated across these assets; cash is excluded from the "
            f"simulation itself (it has no price series) but is folded into the risk-free contribution."
        )

        portfolio_fingerprint = (tuple(sorted(weights.items())), tuple(sorted((k, v) for k, v in tickers_map.items())), fr_years)

        if st.button("Generate Efficient Frontier"):
            period = "1y" if fr_years <= 1 else ("5y" if fr_years <= 5 else "10y")
            tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}
            try:
                with st.spinner("Pulling price history and simulating portfolios..."):
                    prices = fetch_price_history(tuple(tickers_used.values()), period=period)
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
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]),
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
# PAGE: SENSITIVITY INDEX
# =======================================================================
elif page == "Sensitivity Index":
    st.subheader("Sensitivity Index")
    st.caption("How sensitive your portfolio and each asset class are to interest rate moves, broad market moves, and oil price moves — estimated via regression against 3 years of daily data.")

    st.markdown("**Central Bank Policy Rates**")
    rates = fetch_central_bank_rates()
    if rates:
        cols = st.columns(len(rates))
        for col, (name, (rate, as_of_rate)) in zip(cols, rates.items()):
            with col:
                st.markdown(f"""
                    <div class="kpi-box">
                        <div class="label">{name}</div>
                        <div class="value">{rate:.2f}%</div>
                        <div class="asof">as of {as_of_rate}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("Central bank rate data unavailable right now.")

    st.divider()

    selection = select_portfolio_for_page("sensitivity")
    if selection:
        weights = selection["weights"]
        tickers_map = selection["tickers_map"]
        labels_map = selection["labels_map"]
        tickers_used = {k: tickers_map[k] for k in weights if tickers_map.get(k) is not None}

        if st.button("Calculate Sensitivity"):
            with st.spinner("Running regressions against rate, market, and oil moves..."):
                sens = compute_sensitivity(tuple(tickers_used.values()))
            st.session_state["_sensitivity"] = sens

        sens = st.session_state.get("_sensitivity")
        if sens:
            rows = []
            port_rate_beta, port_market_beta, port_oil_beta = 0.0, 0.0, 0.0
            for k, ticker in tickers_used.items():
                s = sens.get(ticker, {})
                rb, mb, ob = s.get("rate_beta"), s.get("market_beta"), s.get("oil_beta")
                rows.append({
                    "Asset Class": labels_map.get(k, k), "Ticker": ticker,
                    "Rate Sensitivity (per +1% in 10yr yield)": f"{rb*100:+.2f}%" if rb is not None else "N/A",
                    "Market Beta (vs S&P 500)": f"{mb:.2f}" if mb is not None else "N/A",
                    "Oil Beta (vs WTI Crude)": f"{ob:.2f}" if ob is not None else "N/A",
                })
                if rb is not None:
                    port_rate_beta += weights[k] * rb
                if mb is not None:
                    port_market_beta += weights[k] * mb
                if ob is not None:
                    port_oil_beta += weights[k] * ob

            st.markdown("**Portfolio-Level Sensitivity**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rate Sensitivity", f"{port_rate_beta*100:+.2f}%", help="Estimated % move in your portfolio for a +1% move in the U.S. 10-year Treasury yield.")
            c2.metric("Market Beta", f"{port_market_beta:.2f}", help="Estimated sensitivity to broad S&P 500 moves. 1.0 = moves in line with the market.")
            c3.metric("Oil Beta", f"{port_oil_beta:.2f}", help="Estimated sensitivity to WTI crude oil price moves.")

            st.divider()
            st.markdown("**By Asset Class**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.caption("Rate sensitivity is the slope of asset daily returns regressed against daily changes in the U.S. 10-year Treasury yield (^TNX) — a higher negative number means the asset tends to fall more when yields rise. Market and oil betas are regressed the same way against the S&P 500 and WTI crude, respectively.")


# =======================================================================
# PAGE: REBALANCING SIMULATOR
# =======================================================================
elif page == "Rebalancing Simulator":
    st.subheader("Rebalancing Simulator")
    st.caption("Pick a saved portfolio to see how its weights would have drifted since you saved it, and what trades would bring it back to target.")

    with st.expander("How this works"):
        st.markdown("""
        When you save a portfolio, Wealthscope records its target weights, the dollar amount, and the save date.
        This tool then:
        1. Pulls actual historical prices for each holding from the save date to today
        2. Calculates how many "shares" your dollar amount would have bought at the save-date price
        3. Revalues those shares at today's price to see your current dollar value per holding
        4. Compares the resulting current weights to your original targets — the difference is drift
        5. Calculates the dollar trade needed per holding to bring weights back to target

        **Why it might say no data is available:** yfinance (the price data source) needs at least one full
        trading day to have closed between your save date and today. If you just saved a portfolio, come back
        tomorrow.
        """)

    portfolios = load_saved_portfolios(username)
    if not portfolios:
        st.info("No saved portfolios yet — save one from the Calculator page first.")
    else:
        options = [f"{p['name']} (saved {p['date_saved']})" for p in portfolios]
        selected_idx = st.selectbox("Choose a saved portfolio", range(len(options)), format_func=lambda i: options[i])
        chosen = portfolios[selected_idx]

        tickers_map = dict(TICKERS)
        labels_map = dict(ASSET_LABELS)
        for key, v in chosen.get("custom_tickers", {}).items():
            tickers_map[key] = v["ticker"]
            labels_map[key] = f"{v['name']} ({v['ticker']})"

        drift_threshold = st.slider("Rebalancing threshold (%)", 1, 20, 5)

        if st.button("Simulate Drift"):
            with st.spinner("Pulling price history since the save date..."):
                result = simulate_rebalancing(chosen["weights"], tickers_map, chosen["amount"], chosen["date_saved"])
            st.session_state["_rebal_result"] = result

        result = st.session_state.get("_rebal_result")
        if result:
            if "error" in result:
                st.error(f"Couldn't simulate drift: {result['error']}")
            else:
                st.caption(f"Simulated from {result['start_date']} to today. Current total value: ${result['total_current']:,.2f} (started at ${chosen['amount']:,.2f}).")

                rows = []
                any_breach = False
                for k in chosen["weights"]:
                    target = chosen["weights"][k]
                    current = result["current_weights"].get(k, 0.0)
                    drift = result["drift"].get(k, 0.0)
                    breach = abs(drift) * 100 > drift_threshold
                    any_breach = any_breach or breach
                    rows.append({
                        "Asset Class": labels_map.get(k, k),
                        "Target Weight": f"{target*100:.1f}%",
                        "Current Weight": f"{current*100:.1f}%",
                        "Drift": f"{drift*100:+.1f}%",
                        "Suggested Trade": f"{'Buy' if result['trades_needed'][k] > 0 else 'Sell'} ${abs(result['trades_needed'][k]):,.0f}",
                        "Breach?": "Yes" if breach else "No",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if any_breach:
                    st.warning(f"One or more asset classes have drifted more than {drift_threshold}% from target — rebalancing is suggested.")
                else:
                    st.success(f"No asset class has drifted more than {drift_threshold}% from target — no rebalancing needed yet.")


# =======================================================================
# PAGE: QUARTERLY VIEWS
# =======================================================================
elif page == "Quarterly Views":
    st.subheader("Quarterly Portfolio Views")
    st.caption("Shows how your selected portfolio's allocation would have looked at the start of each quarter of a chosen year, based on historical market conditions at that time.")

    selection = select_portfolio_for_page("quarterly")
    if selection:
        base_weights = selection["weights"]
        horizon = selection["horizon"]
        labels_map = selection["labels_map"]

        today = date.today()
        year_options = [today.year, today.year - 1, today.year - 2]
        selected_year = st.selectbox("Year", year_options, index=0)

        quarter_starts = [date(selected_year, m, 1) for m in (1, 4, 7, 10) if date(selected_year, m, 1) <= today]

        if selected_year == today.year and len(quarter_starts) < 4:
            st.caption(f"{selected_year} is the current year, so only the {len(quarter_starts)} quarter(s) that have already started are shown — future quarters don't have market data yet.")

        if st.button("Generate Quarterly Views"):
            with st.spinner("Pulling historical market conditions for each quarter..."):
                quarterly_data = []
                for q_date in quarter_starts:
                    hist_score = historical_market_condition_score(q_date.strftime("%Y-%m-%d"))
                    w = apply_tactical_tilt(base_weights, horizon, hist_score)
                    quarterly_data.append((q_date, hist_score, w))
                st.session_state["_quarterly"] = quarterly_data

        quarterly_data = st.session_state.get("_quarterly")
        if quarterly_data:
            cols = st.columns(len(quarterly_data))
            for col, (q_date, hist_score, w) in zip(cols, quarterly_data):
                with col:
                    quarter_num = (q_date.month - 1) // 3 + 1
                    st.markdown(f"**Q{quarter_num} {q_date.year}**")
                    st.caption(f"Market score: {hist_score:+.2f}" if hist_score is not None else "Market score: N/A")
                    fig = go.Figure(data=[go.Pie(labels=[labels_map.get(k, k) for k in w], values=[v * 100 for v in w.values()],
                                                  hole=0.45, textinfo="percent")])
                    fig.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10),
                                       paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]))
                    st.plotly_chart(fig, use_container_width=True, key=f"q_pie_{q_date}")

            st.divider()
            st.markdown("**Weight Evolution by Asset Class**")
            evo_df = pd.DataFrame({
                f"Q{(q_date.month - 1) // 3 + 1} {q_date.year}": {labels_map.get(k, k): round(v * 100, 1) for k, v in w.items()}
                for q_date, _, w in quarterly_data
            }).T
            st.dataframe(evo_df, use_container_width=True)


# =======================================================================
# PAGE: COMPARE PROFILES
# =======================================================================
elif page == "Compare Profiles":
    st.subheader("Compare Profiles")
    compare_mode = st.radio("What do you want to compare?", ["Standard Risk Profiles", "My Saved Portfolios"], horizontal=True)

    if compare_mode == "Standard Risk Profiles":
        compare_amount = st.number_input("Amount for comparison ($)", min_value=1.0, max_value=10_000_000.0, value=20000.0, step=500.0, key="compare_amount")
        compare_horizon = st.number_input("Horizon for comparison (years)", min_value=1, max_value=50, value=10, key="compare_horizon")

        score, as_of, _ = market_condition_score(str(date.today()))
        cols = st.columns(3)
        for i, profile in enumerate(RiskProfile):
            w = build_portfolio(profile, compare_horizon, score)
            with cols[i]:
                st.markdown(f"**{profile.value}**")
                fig = go.Figure(data=[go.Pie(labels=[ASSET_LABELS[k] for k in w], values=[v * 100 for v in w.values()],
                                              hole=0.45, textinfo="percent")])
                fig.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10, l=10, r=10),
                                   paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]))
                st.plotly_chart(fig, use_container_width=True, key=f"compare_pie_{i}")
                df = pd.DataFrame({"Asset": [ASSET_LABELS[k] for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()],
                                    "$": [f"${compare_amount*v:,.0f}" for v in w.values()]})
                st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        portfolios = load_saved_portfolios(username)
        if not portfolios:
            st.info("No saved portfolios yet — save one from the Calculator page first.")
        else:
            names = [p["name"] for p in portfolios]
            selected_names = st.multiselect("Choose up to 3 saved portfolios", names, max_selections=3)

            # Must run before any compare_slider_* widgets below are instantiated this
            # run — same Streamlit restriction as the expected-returns reset.
            for p in portfolios:
                flag_key = f"_reset_compare_{p['name']}"
                if st.session_state.get(flag_key):
                    for k in p["weights"]:
                        st.session_state[f"compare_slider_{p['name']}_{k}"] = int(round(p["weights"][k] * 100))
                    st.session_state[flag_key] = False

            if not selected_names:
                st.caption("Pick at least one saved portfolio above to compare.")
            else:
                selected_portfolios = [p for p in portfolios if p["name"] in selected_names]
                st.caption("Adjust the sliders under any portfolio to see how a change would affect it — this is a live \"what-if\" view only and never changes your actual saved portfolio.")
                cols = st.columns(len(selected_portfolios))
                for i, p in enumerate(selected_portfolios):
                    original_w = p["weights"]
                    p_labels = dict(ASSET_LABELS)
                    for key, v in p.get("custom_tickers", {}).items():
                        p_labels[key] = f"{v['name']} ({v['ticker']})"

                    with cols[i]:
                        st.markdown(f"**{p['name']}**")
                        st.caption(f"{p['risk_profile']}, {p['horizon']}yr, ${p['amount']:,.0f}")

                        with st.expander("Adjust weights"):
                            edited = {}
                            for k in original_w:
                                edited[k] = st.slider(
                                    p_labels.get(k, k), 0, 100, int(round(original_w[k] * 100)),
                                    key=f"compare_slider_{p['name']}_{k}",
                                )
                            raw_total = sum(edited.values())
                            w = {k: v / raw_total for k, v in edited.items()} if raw_total > 0 else original_w
                            changed = any(abs(w[k] - original_w[k]) > 0.001 for k in original_w)
                            if changed:
                                st.caption("Showing your edited weights (normalized to 100%).")
                                if st.button("Reset to saved weights", key=f"compare_reset_{p['name']}"):
                                    st.session_state[f"_reset_compare_{p['name']}"] = True
                                    st.rerun()
                            else:
                                w = original_w

                        exp_ret = expected_return({k: v for k, v in w.items() if k in st.session_state.custom_expected_returns},
                                                   st.session_state.custom_expected_returns)
                        st.metric("Illustrative Expected Return", f"{exp_ret*100:.2f}%")

                        fig = go.Figure(data=[go.Pie(labels=[p_labels.get(k, k) for k in w], values=[v * 100 for v in w.values()],
                                                      hole=0.45, textinfo="percent")])
                        fig.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10, l=10, r=10),
                                           paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]))
                        st.plotly_chart(fig, use_container_width=True, key=f"saved_compare_pie_{i}")
                        df = pd.DataFrame({"Asset": [p_labels.get(k, k) for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()],
                                            "$": [f"${p['amount']*v:,.0f}" for v in w.values()]})
                        st.dataframe(df, use_container_width=True, hide_index=True)


# =======================================================================
# PAGE: MARKET NEWS
# =======================================================================
elif page == "Market News":
    st.subheader("Recent News by Asset Class")
    selection = select_portfolio_for_page("news")
    if selection:
        tickers_map = selection["tickers_map"]
        labels_map = selection["labels_map"]
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
# PAGE: DAILY BRIEFING
# =======================================================================
elif page == "Daily Briefing":
    st.subheader("Daily Briefing")
    st.caption("Top headlines from your focus areas, refreshed every few hours — pulled from dedicated topic feeds, not your held portfolio.")

    briefing = fetch_daily_briefing()

    for sector, headlines in briefing.items():
        st.markdown(f"### {sector}")
        if not headlines:
            st.caption("No recent headlines available for this sector right now.")
        else:
            for h in headlines:
                pub = f" — *{h['publisher']}*" if h["publisher"] else ""
                if h["link"]:
                    st.markdown(f"- [{h['title']}]({h['link']}){pub}")
                else:
                    st.markdown(f"- {h['title']}{pub}")
        st.divider()

    st.caption(
        "Headlines are sourced from dedicated topic feeds: Investment Product Advisory (ETF Analysis & Opinion), "
        "FX Market (Forex News), Metals & Mining (Metals Analysis), Bond Market (Bonds Analysis & Opinion) — "
        "each feed is specific to that theme, not a stock-price proxy."
    )


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
                p_labels = dict(ASSET_LABELS)
                for key, v in p.get("custom_tickers", {}).items():
                    p_labels[key] = f"{v['name']} ({v['ticker']})"
                df = pd.DataFrame({"Asset": [p_labels.get(k, k) for k in w], "Weight": [f"{v*100:.1f}%" for v in w.values()],
                                    "$": [f"${p['amount']*v:,.0f}" for v in w.values()]})
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.button("Delete", key=f"delete_{i}"):
                    delete_portfolio(username, i)
                    st.rerun()


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
                                   paper_bgcolor="rgba(0,0,0,0)", font=dict(color=THEME_PALETTES[st.session_state.theme]["text"]), height=280)
                st.plotly_chart(fig, use_container_width=True, key=f"strategy_pie_{profile.value}")
            with c2:
                st.markdown(f"**Philosophy:** {notes['philosophy']}")
                st.markdown(f"**Expected Volatility:** {notes['expected_volatility']}")
                st.markdown(f"**Best suited for:** {notes['who_its_for']}")

    st.divider()
    st.markdown("""
        **How the tactical tilt works:** Each profile's weights above are the *strategic* baseline.
        On the Calculator page, two things nudge the final allocation within a ±15% range:
        - **Horizon** — shifts every 2 years, from most defensive (0-2yr) to most aggressive (20+yr)
        - **Market conditions** — calculated from the previous trading day's VIX level and S&P 500 momentum at market open
    """)
