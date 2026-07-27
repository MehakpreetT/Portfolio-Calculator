# PortPicker

A live portfolio construction web app that builds a target asset allocation from a user's risk profile, investment horizon, and current market conditions — then lets them backtest it, stress-test it, analyze its sensitivity to macro factors, simulate rebalancing, and track it over time. Supports user accounts so each person's portfolios stay private and saved.

**Live app:** https://portfolio-calculator-c4n2sqxe8zkfqttgtzpbg6.streamlit.app/

Built with Python and Streamlit.

---

## What it does

PortPicker starts from a strategic asset-mix baseline (Conservative / Neutral / Growth), then applies two tactical tilts within a ±15% range:

- **Investment horizon** — steps every 2 years (0-2yr through 20+yr, in ten 2-year buckets), from most defensive to most aggressive
- **Market conditions** — a composite score built from the prior trading day's VIX level and S&P 500 momentum at market open

From there, a user can override weights manually, add any North American ticker as a custom holding, set a stop-loss threshold, and run the resulting portfolio through several analysis tools — each of which lets you choose whether to analyze the portfolio currently active on the Calculator or any previously saved one.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Live market condition score, all G10 central bank policy rates, G10 currency rates (refreshed every 15 minutes), and a snapshot of your most recent saved portfolio |
| **Risk Questionnaire** | 5 plain-language questions that suggest a risk profile, with a one-click handoff to the Calculator |
| **Calculator** | Builds the recommended allocation with an illustrative expected-return estimate, supports custom weights, custom tickers, a stop-loss threshold, data validation (only shown when something's actually wrong), and CSV/PDF export |
| **Backtest & Risk** | Historical performance (1/5/10yr) vs. a 60/40 benchmark, with Sharpe ratio and max drawdown |
| **Stress Testing** | Applies actual historical returns from 4 crisis periods (2008 GFC, 2020 COVID crash, 2022 rate-hike selloff, 2015-16 oil crash) to the selected portfolio |
| **Efficient Frontier** | Monte Carlo simulation (3,000 random portfolios) plus a solved max-Sharpe point, with the selected portfolio's actual position plotted on the same chart — always reflects whichever portfolio is chosen, never stale |
| **Sensitivity Index** | Regression-based sensitivity to U.S. 10-year yield changes, S&P 500 moves, and oil price moves — per asset class and portfolio-wide |
| **Rebalancing Simulator** | Shows how a saved portfolio's weights have actually drifted since it was saved (using real price history), flags threshold breaches, and suggests trades to rebalance |
| **Quarterly Views** | What the selected portfolio would have looked like at the start of each of the last 4 quarters, tailored to that specific portfolio (including custom tickers and manual overrides), using historical market data for each date |
| **Compare Profiles** | Conservative / Neutral / Growth allocations side-by-side |
| **Market News** | Recent headlines per asset class for the selected portfolio |
| **Saved Portfolios** | Per-account saved strategies (up to 10), including custom ticker holdings |
| **Portfolio Education** | Plain-language explanation of the 5-step portfolio construction process and what each asset class is and why it's used |
| **Strategy Guide** | Philosophy, expected volatility, typical historical drawdown range, time-horizon fit, and rebalancing cadence for each risk profile |

**Portfolio selector:** every analysis page (Backtest, Stress Testing, Efficient Frontier, Sensitivity Index, Quarterly Views, Market News) has a dropdown to choose between the portfolio currently active on the Calculator and any of your saved portfolios — so you can compare different strategies without rebuilding them each time.

## Methodology notes

- **Asset classes:** Cash, Fixed Income, Canadian Equity, U.S. Equity, International Equity, Emerging Markets, Gold/Commodities, and REITs — represented by iShares/Vanguard ETFs (e.g. XBB.TO, VFV.TO, XEF.TO).
- **Expected return figures** shown on the Calculator and Dashboard are illustrative, based on long-run capital market assumptions per asset class — not a forecast.
- **Sharpe ratio** calculations assume a 3% annualized risk-free rate.
- **Sensitivity betas** (rate/market/oil) are OLS regression slopes of each asset's daily returns against daily changes in the U.S. 10-year Treasury yield, S&P 500 returns, and WTI crude returns, computed independently per ticker so assets with different listing histories don't distort each other's results.
- Investment amount is capped at $10,000,000 and horizon at 50 years.
- All backtests, stress tests, and the rebalancing simulator use actual historical price data pulled live via `yfinance` — nothing is simulated except the Efficient Frontier's random portfolio cloud.

## Tech stack

- **Frontend/framework:** Streamlit, with a left sidebar navigation (`streamlit-option-menu`) and a custom dark theme
- **Data:** yfinance (prices, news), pandas-datareader + FRED (central bank rates)
- **Analysis:** pandas, numpy, scipy (portfolio optimization)
- **Auth:** streamlit-authenticator (bcrypt-hashed passwords, signed session cookies via `st.secrets`)
- **Visualization:** Plotly
- **Export:** fpdf2 (PDF), pandas (CSV)

## Running locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` in the project root with:

```toml
cookie_key = "your-own-random-secret-here"
```

Then run:

```bash
streamlit run streamlit_app.py
```

A test account is auto-seeded on first run (and self-heals if `users.yaml` already exists from an earlier version without it):
- **Username:** `testuser`
- **Password:** `test1234`

## Deploying

1. Push this repo to GitHub (the `.gitignore` already excludes `secrets.toml`, `users.yaml`, and saved portfolio data — never commit these)
2. Deploy on [Streamlit Community Cloud](https://share.streamlit.io), pointing at `streamlit_app.py`
3. In the app's dashboard, go to **Settings → Secrets** and add the same `cookie_key` value
4. Streamlit Cloud auto-redeploys on every push to the connected branch

## Known limitations

- **Storage is local to the container.** User accounts (`users.yaml`) and saved portfolios (`saved_portfolios/`) are stored as flat files, which reset when the app restarts or sleeps on Streamlit Cloud's free tier. Fine for demos; not durable production storage.
- **Illustrative, not investment advice.** Expected return assumptions, the market condition score, and the tactical tilt logic are simplified models built for demonstration and learning purposes.
- **Central bank rate coverage varies.** Some G10 policy rate series update monthly rather than daily (a function of how often those central banks actually change rates and how often the underlying FRED series is republished) — the "as of" date on each card shows exactly how current each figure is.
