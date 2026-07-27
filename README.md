# PortPicker

A live portfolio construction web app that builds a target asset allocation from a user's risk profile, investment horizon, and current market conditions — then lets them backtest it, stress-test it, analyze its sensitivity to macro factors, and track it over time.

**Live app:** https://portfolio-calculator-c4n2sqxe8zkfqttgtzpbg6.streamlit.app/

Built with Python and Streamlit.

---

## What it does

PortPicker starts from a strategic asset-mix baseline (Conservative / Neutral / Growth), then applies two tactical tilts within a ±15% range:

- **Investment horizon** — steps every 2 years, from most defensive (0-2yr) to most aggressive (20+yr)
- **Market conditions** — a composite score built from the prior trading day's VIX level and S&P 500 momentum at market open

From there, a user can override weights manually, add any North American ticker as a custom holding, and run the resulting portfolio through several analysis tools.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Live market condition score, central bank policy rates (Fed / BoC / BoE), and a snapshot of your most recent saved portfolio |
| **Risk Questionnaire** | 5 plain-language questions that suggest a risk profile |
| **Calculator** | Builds the recommended allocation, supports manual weight overrides, custom tickers, a stop-loss threshold, data validation, and CSV/PDF export |
| **Backtest & Risk** | Historical performance (1/5/10yr) vs. a 60/40 benchmark, with Sharpe ratio and max drawdown |
| **Stress Testing** | Applies actual historical returns from 4 crisis periods (2008 GFC, 2020 COVID crash, 2022 rate-hike selloff, 2015-16 oil crash) to the current portfolio |
| **Efficient Frontier** | Monte Carlo simulation (3,000 random portfolios) plus a solved max-Sharpe point, with the user's actual portfolio plotted on the same chart |
| **Sensitivity Index** | Regression-based sensitivity to U.S. 10-year yield changes, S&P 500 moves, and oil price moves — per asset class and portfolio-wide |
| **Rebalancing Simulator** | Shows how a saved portfolio's weights have drifted since it was saved, and what trades would bring it back to target |
| **Quarterly Views** | What the current risk profile/horizon would have recommended at the start of each of the last 4 quarters, using historical market data |
| **Compare Profiles** | Conservative / Neutral / Growth allocations side-by-side |
| **Market News** | Recent headlines per asset class |
| **Saved Portfolios** | Per-account saved strategies |
| **Portfolio Education** | Plain-language explanation of portfolio construction and what each asset class is |
| **Strategy Guide** | Philosophy and expected volatility behind each risk profile |

## Methodology notes

- **Asset classes:** Cash, Fixed Income, Canadian Equity, U.S. Equity, International Equity, Emerging Markets, Gold/Commodities, and REITs — represented by iShares/Vanguard ETFs (e.g. XBB.TO, VFV.TO, XEF.TO).
- **Expected return figures** shown on the Calculator are illustrative, based on long-run capital market assumptions per asset class — not a forecast.
- **Sharpe ratio** calculations assume a 3% annualized risk-free rate.
- All backtests and stress tests use actual historical price data pulled live via `yfinance` — nothing is simulated except the Efficient Frontier's random portfolio cloud.

## Tech stack

- **Frontend/framework:** Streamlit
- **Data:** yfinance (prices, news), pandas-datareader + FRED (central bank rates)
- **Analysis:** pandas, numpy, scipy (portfolio optimization)
- **Auth:** streamlit-authenticator (bcrypt-hashed passwords, signed session cookies)
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

A test account is auto-seeded on first run:
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
