#!/usr/bin/env python3
import argparse
import os
from typing import List

import pandas as pd
import yfinance as yf
import cufflinks as cf
from plotly import offline as pyo
import plotly.graph_objects as go


def parse_periods(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def build_qf(
    df: pd.DataFrame,
    title: str,
    smas: List[int],
    emas: List[int],
    add_volume: bool,
    rsi_period: int = 0,
    macd_fast: int = 0,
    macd_slow: int = 0,
    macd_signal: int = 0,
    bb_period: int = 0,
    bb_std: float = 2.0,
):
    qf = cf.QuantFig(df, title=title, legend='top', up_color='green', down_color='red')
    if add_volume and 'Volume' in df.columns:
        qf.add_volume()
    for p in smas:
        qf.add_sma(periods=p)
    for p in emas:
        qf.add_ema(periods=p)
    if rsi_period and rsi_period > 0:
        qf.add_rsi(periods=int(rsi_period))
    if macd_fast and macd_slow and macd_signal:
        qf.add_macd(fast_period=int(macd_fast), slow_period=int(macd_slow), signal_period=int(macd_signal))
    if bb_period and bb_period > 0:
        qf.add_bollinger_bands(periods=int(bb_period), boll_std=float(bb_std))
    return qf


def main():
    p = argparse.ArgumentParser(description='Generate a candlestick chart with optional indicators (cufflinks/plotly).')
    p.add_argument('--ticker', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--title', default=None)
    p.add_argument('--smas', help='Comma-separated SMA periods, e.g., 20,50')
    p.add_argument('--emas', help='Comma-separated EMA periods, e.g., 12,26')
    p.add_argument('--no-volume', action='store_true', help='Do not add volume pane')
    p.add_argument('--rsi', type=int, default=0, help='Add RSI with given period (e.g., 14)')
    p.add_argument('--macd', type=str, help='Add MACD with fast,slow,signal periods (e.g., 12,26,9)')
    p.add_argument('--bbands', type=str, help='Add Bollinger Bands as period,std (e.g., 20,2)')
    p.add_argument('--output', required=True, help='Output HTML file for the interactive chart')
    args = p.parse_args()

    df = yf.download(args.ticker, start=args.start, end=args.end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise SystemExit(f'No data for {args.ticker}')

    # Ensure proper columns and naming expected by QuantFig
    if isinstance(df.columns, pd.MultiIndex):
        # If only one ticker present, drop the ticker level
        tickers = df.columns.get_level_values(1).unique().tolist()
        if len(tickers) == 1:
            df = df.droplevel(1, axis=1)
        else:
            # Select the requested ticker
            if args.ticker in tickers:
                df = df.xs(args.ticker, axis=1, level=1)
            else:
                # Fallback: pick first ticker
                df = df.xs(tickers[0], axis=1, level=1)
    # Now pick OHLCV
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    title = args.title or f'{args.ticker} Candlestick'
    smas = parse_periods(args.smas)
    emas = parse_periods(args.emas)
    add_volume = not args.no_volume

    # Set up plotly offline for safe export
    # Parse extra indicators
    rsi_period = int(args.rsi) if args.rsi else 0
    macd_fast = macd_slow = macd_signal = 0
    if args.macd:
        try:
            macd_fast, macd_slow, macd_signal = [int(x.strip()) for x in args.macd.split(',')]
        except Exception:
            print('Invalid --macd value; expected "fast,slow,signal". Skipping MACD.')
    bb_period = 0
    bb_std = 2.0
    if args.bbands:
        try:
            parts = [x.strip() for x in args.bbands.split(',')]
            bb_period = int(parts[0])
            if len(parts) > 1:
                bb_std = float(parts[1])
        except Exception:
            print('Invalid --bbands value; expected "period,std". Skipping Bollinger Bands.')

    # Offline plotting (HTML export)
    cf.go_offline()

    try:
        qf = build_qf(
            df,
            title=title,
            smas=smas,
            emas=emas,
            add_volume=add_volume,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_period=bb_period,
            bb_std=bb_std,
        )
        fig = qf.iplot(asFigure=True)
    except Exception as e:
        # Fallback to pure plotly if cufflinks has compatibility issues
        print(f"Cufflinks rendering failed ({e}). Falling back to pure Plotly.")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC'
        ))
        # Overlays
        for p in smas:
            sma = df['Close'].rolling(window=p, min_periods=p).mean()
            fig.add_trace(go.Scatter(x=df.index, y=sma, mode='lines', name=f'SMA {p}'))
        for p in emas:
            ema = df['Close'].ewm(span=p, adjust=False).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ema, mode='lines', name=f'EMA {p}'))
        # Note: RSI/MACD/BBands overlays are only available in the cufflinks path.
        # Volume on secondary y-axis as bars
        if add_volume and 'Volume' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.3, marker_color='gray', yaxis='y2'))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume', rangemode='tozero', scaleanchor=None, position=1.0))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pyo.plot(fig, filename=args.output, auto_open=False)
    print(f'Saved candlestick chart to {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
