#!/usr/bin/env python3
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Data fetching implementations
# -----------------------------

def fetch_from_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance using yfinance.

    Returns a DataFrame with index as date and a standardized 'adj_close' column.
    Handles both single-level and MultiIndex columns.
    """
    import yfinance as yf

    # Use auto_adjust=True so 'Close' reflects adjusted close
    df = yf.download(tickers=ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned from Yahoo for ticker {ticker}.")

    def pick_field(available, preferences):
        lookup = {str(a).strip().lower(): a for a in available}
        for name in preferences:
            key = str(name).strip().lower()
            if key in lookup:
                return lookup[key]
        # fallback: anything containing 'close'
        for key, original in lookup.items():
            if 'close' in key:
                return original
        return None

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        available_fields = list(pd.unique(level0))
        selected_field = pick_field(available_fields, ["Adj Close", "Adjusted Close", "Close"])
        if selected_field is None:
            raise ValueError(
                "Could not find a price field (Adj Close/Close) in Yahoo data columns: "
                + ", ".join([str(c) for c in df.columns])
            )
        tickers_present = list(pd.unique(level1))
        if not tickers_present:
            raise ValueError("Unexpected Yahoo data format: no tickers in MultiIndex columns.")
        # Single ticker download should have one ticker; pick the first
        selected_ticker = tickers_present[0]
        series = df[(selected_field, selected_ticker)].copy()
    else:
        available = list(df.columns)
        selected_field = pick_field(available, ["Adj Close", "Adjusted Close", "Close"])
        if selected_field is None:
            raise ValueError(
                "Neither Adjusted Close nor Close found in Yahoo data columns: "
                + ", ".join([str(c) for c in df.columns])
            )
        series = df[selected_field].copy()

    out = pd.DataFrame({'adj_close': series})
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    return out


def fetch_from_quandl(dataset: str, ticker: str, start: str, end: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical data from Quandl.

    dataset: e.g., 'EOD', 'FSE', etc.
    Returns a DataFrame with 'adj_close' column if available, else 'close'.
    """
    if not dataset:
        raise ValueError("Quandl requires --dataset (e.g., 'EOD').")

    import quandl

    key = api_key or os.environ.get('QUANDL_API_KEY')
    if not key:
        raise ValueError("Missing Quandl API key. Set QUANDL_API_KEY or pass --quandl-api-key.")
    quandl.ApiConfig.api_key = key

    qcode = f"{dataset}/{ticker}"
    df = quandl.get(dataset=qcode, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError(f"No data returned from Quandl for code {qcode}.")

    # Try common adjusted/close column names in order of preference
    candidates = [
        'Adj. Close', 'Adj Close', 'Adj_Close', 'Adjusted Close',
        'adj_close', 'adj. close', 'adj close',
    ]
    fallback = ['Close', 'close']

    selected = None
    for c in candidates:
        if c in df.columns:
            selected = c
            break
    if selected is None:
        for c in fallback:
            if c in df.columns:
                selected = c
                break
    if selected is None:
        raise ValueError("Could not find an adjusted or close price column in Quandl response.")

    out = pd.DataFrame({'adj_close': df[selected].copy()})
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    return out


def fetch_from_intrinio(ticker: str, start: str, end: str, frequency: str = 'daily', page_size: int = 10000, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical prices from Intrinio using intrinio_sdk.

    Returns a DataFrame with 'adj_close' column if available, else 'close'.
    """
    import intrinio_sdk

    key = api_key or os.environ.get('INTRINIO_API_KEY')
    if not key:
        raise ValueError("Missing Intrinio API key. Set INTRINIO_API_KEY or pass --intrinio-api-key.")

    intrinio_sdk.ApiClient().configuration.api_key['api_key'] = key

    security_api = intrinio_sdk.SecurityApi()
    resp = security_api.get_security_stock_prices(
        identifier=ticker,
        start_date=start,
        end_date=end,
        frequency=frequency,
        page_size=page_size,
    )

    prices = resp.stock_prices or []
    if not prices:
        raise ValueError(f"No data returned from Intrinio for {ticker}.")

    # Convert to DataFrame
    records = []
    for p in prices:
        records.append({
            'date': pd.to_datetime(getattr(p, 'date', None)),
            'close': getattr(p, 'close', None),
            'adj_close': getattr(p, 'adj_close', None),
        })
    df = pd.DataFrame.from_records(records).set_index('date').sort_index()

    if 'adj_close' in df.columns and df['adj_close'].notna().any():
        series = df['adj_close']
    else:
        series = df['close']
        if series.isna().all():
            raise ValueError("Intrinio response lacks both adj_close and close values.")
    out = pd.DataFrame({'adj_close': series})
    return out


# -----------------------------
# Analysis helpers
# -----------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple and log returns based on 'adj_close'."""
    df = prices.copy()
    if 'adj_close' not in df.columns:
        raise ValueError("Input DataFrame must contain 'adj_close'.")

    df['simple_rtn'] = df['adj_close'].pct_change()
    df['log_rtn'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    return df


def flag_outliers(df: pd.DataFrame, window: int = 21, z: float = 3.0) -> pd.DataFrame:
    """Flag outliers in simple returns using rolling mean/std and z-score threshold.

    Adds columns: 'rtn_ma', 'rtn_sigma', 'upper_band', 'lower_band', 'is_outlier'.
    """
    out = df.copy()
    if 'simple_rtn' not in out.columns:
        raise ValueError("DataFrame must contain 'simple_rtn' column.")

    out['rtn_ma'] = out['simple_rtn'].rolling(window=window, min_periods=window).mean()
    out['rtn_sigma'] = out['simple_rtn'].rolling(window=window, min_periods=window).std(ddof=0)
    out['upper_band'] = out['rtn_ma'] + z * out['rtn_sigma']
    out['lower_band'] = out['rtn_ma'] - z * out['rtn_sigma']

    def is_outlier(x, mean, sigma):
        if pd.isna(mean) or pd.isna(sigma):
            return 0
        return 1 if (x > mean + z * sigma) or (x < mean - z * sigma) else 0

    out['is_outlier'] = [
        is_outlier(x, m, s) for x, m, s in zip(out['simple_rtn'], out['rtn_ma'], out['rtn_sigma'])
    ]
    return out


# -----------------------------
# CLI
# -----------------------------

@dataclass
class Args:
    source: str
    ticker: Optional[str]
    start: str
    end: str
    dataset: Optional[str]
    frequency: str
    window: int
    quandl_api_key: Optional[str]
    intrinio_api_key: Optional[str]
    output: Optional[str]
    tickers: Optional[str]
    tickers_file: Optional[str]
    plot: bool
    plot_output: Optional[str]


def parse_args(argv=None) -> Args:
    p = argparse.ArgumentParser(description="Fetch prices, compute returns, and flag outliers (3σ)")
    p.add_argument('--source', required=True, choices=['yahoo', 'quandl', 'intrinio'], help='Data source')
    p.add_argument('--ticker', required=False, help='Ticker/symbol, e.g., AAPL (ignored if --tickers/--tickers-file provided)')
    p.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    p.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    p.add_argument('--dataset', help='Quandl dataset prefix, e.g., EOD (required for --source quandl)')
    p.add_argument('--frequency', default='daily', help='Intrinio frequency (default: daily)')
    p.add_argument('--window', type=int, default=21, help='Rolling window for outlier detection (default: 21)')
    p.add_argument('--quandl-api-key', dest='quandl_api_key', help='Quandl API key (overrides env)')
    p.add_argument('--intrinio-api-key', dest='intrinio_api_key', help='Intrinio API key (overrides env)')
    p.add_argument('--output', help='Optional CSV path to save results. For multiple tickers, include {ticker} or a suffix will be added.')
    p.add_argument('--tickers', help='Comma-separated list of tickers for batch processing')
    p.add_argument('--tickers-file', help='Path to a file with one ticker per line')
    p.add_argument('--plot', action='store_true', help='Generate plots (price and returns with outliers)')
    p.add_argument('--plot-output', dest='plot_output', help='Path to save plot image. For multiple tickers, include {ticker} or a suffix will be added. If omitted, plots are shown interactively.')
    ns = p.parse_args(argv)
    return Args(
        source=ns.source,
        ticker=ns.ticker,
        start=ns.start,
        end=ns.end,
        dataset=ns.dataset,
        frequency=ns.frequency,
        window=ns.window,
        quandl_api_key=ns.quandl_api_key,
        intrinio_api_key=ns.intrinio_api_key,
        output=ns.output,
        tickers=ns.tickers,
        tickers_file=ns.tickers_file,
        plot=ns.plot,
        plot_output=ns.plot_output,
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    def format_out_path(base: Optional[str], ticker: str, default_name: str) -> Optional[str]:
        if base is None:
            return None
        base_path = Path(base)
        if '{ticker}' in base_path.as_posix():
            return base_path.as_posix().replace('{ticker}', ticker)
        # append ticker before suffix
        if base_path.suffix:
            return base_path.with_name(base_path.stem + f'_{ticker}' + base_path.suffix).as_posix()
        return (base_path.as_posix() + f'_{ticker}')

    def plot_results(ticker: str, df: pd.DataFrame, save_path: Optional[str]) -> None:
        import matplotlib.pyplot as plt
        fig, (ax_price, ax_rtn) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{ticker} Price and Returns (3σ Outliers)")
        # Price
        ax_price.plot(df.index, df['adj_close'], label='Adj Close', color='tab:blue')
        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')
        # Returns with bands
        ax_rtn.plot(df.index, df['simple_rtn'], label='Simple Return', color='tab:gray')
        if 'upper_band' in df.columns and 'lower_band' in df.columns:
            ax_rtn.fill_between(df.index, df['lower_band'], df['upper_band'], color='tab:orange', alpha=0.2, label='±3σ band')
        # Outliers
        outliers = df[df['is_outlier'] == 1]
        ax_rtn.scatter(outliers.index, outliers['simple_rtn'], color='red', s=12, label='Outlier', zorder=3)
        ax_rtn.set_ylabel('Return')
        ax_rtn.legend(loc='upper left')
        ax_rtn.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot to {os.path.abspath(save_path)}")
        else:
            plt.show()

    # Build list of tickers
    tickers_list = []
    if args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers_list.extend([line.strip() for line in f if line.strip()])
    if args.tickers:
        tickers_list.extend([t.strip() for t in args.tickers.split(',') if t.strip()])
    if not tickers_list:
        if args.ticker:
            tickers_list = [args.ticker]
        else:
            raise ValueError("Please provide --ticker or --tickers or --tickers-file.")

    multi = len(tickers_list) > 1

    for tk in tickers_list:
        # Fetch
        if args.source == 'yahoo':
            prices = fetch_from_yahoo(tk, args.start, args.end)
        elif args.source == 'quandl':
            prices = fetch_from_quandl(args.dataset, tk, args.start, args.end, api_key=args.quandl_api_key)
        else:
            prices = fetch_from_intrinio(tk, args.start, args.end, frequency=args.frequency, api_key=args.intrinio_api_key)

        # Compute returns and outliers
        df = compute_returns(prices)
        df = flag_outliers(df, window=args.window, z=3.0)

        # Output CSV
        if args.output:
            out_path = os.path.abspath(format_out_path(args.output, tk, f"{tk}.csv"))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(df).to_csv(out_path, index=True)
            print(f"Saved results to {out_path}")
        else:
            # Print a concise preview for single ticker; for multi, print brief header
            with pd.option_context('display.max_rows', 8, 'display.width', 120):
                print(f"=== {tk} ===")
                print(df.head(4))
                print("...")
                print(df.tail(4))

        # Plot
        if args.plot:
            plot_path = None
            if args.plot_output:
                plot_path = format_out_path(args.plot_output, tk, f"{tk}.png")
            plot_results(tk, df, plot_path)

    return 0


if __name__ == '__main__':
    sys.exit(main()) 