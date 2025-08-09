## Financial Data Fetch & Analysis

This utility fetches historical price data from Yahoo Finance, Quandl, or Intrinio, computes simple and log returns, and flags outliers using a rolling 3σ method.

### Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API keys (as needed):
- Quandl: export QUANDL_API_KEY=your_key
- Intrinio: export INTRINIO_API_KEY=your_key

### Usage

```bash
python fetch_and_analyze.py --source yahoo --ticker AAPL --start 2024-01-01 --end 2024-06-30 --window 21 --output aapl_analysis.csv
```

- `--source`: `yahoo`, `quandl`, or `intrinio`
- `--ticker`: Symbol (e.g., `AAPL`)
- `--start`, `--end`: YYYY-MM-DD
- `--window`: Rolling window for outlier detection (default: 21)
- `--dataset`: Quandl dataset prefix (e.g., `EOD`) when using `--source quandl`
- `--frequency`: Intrinio price frequency (default: `daily`)
- `--output`: Optional CSV file to save results
 - `--tickers`: Comma-separated tickers for batch mode (e.g., `AAPL,MSFT,GOOG`)
 - `--tickers-file`: File with one ticker per line
 - `--plot`: Generate plots (price and returns with 3σ bands)
 - `--plot-output`: Image path to save plot; include `{ticker}` for batch (e.g., `plots/{ticker}.png`)

### Candlestick chart (cufflinks + plotly)

```bash
python candlestick_chart.py --ticker AAPL --start 2024-01-01 --end 2024-06-30 --smas 20,50 --emas 12,26 --output charts/aapl.html
```

### Backtest with backtrader (SMA cross)

```bash
python backtest_strategy.py --ticker AAPL --start 2024-01-01 --end 2024-06-30 --cash 10000 --pfast 10 --pslow 20
```

### Interactive TA dashboard (Jupyter)

Open the notebook and run all cells:

```bash
jupyter notebook ta_dashboard.ipynb
```

### Streamlit Web App

```bash
streamlit run app.py
```

### Notes
- Yahoo Finance requires no API key.
- Quandl and Intrinio require valid API keys.
- Adjusted close is preferred. If not available, close price is used as a fallback. 