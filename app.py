#!/usr/bin/env python3
import os
import io
import datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fetch_and_analyze import (
    fetch_from_yahoo,
    fetch_from_quandl,
    fetch_from_intrinio,
    compute_returns,
    flag_outliers,
)

import yfinance as yf
import backtrader as bt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    import pmdarima as pm
except Exception:  # pragma: no cover
    pm = None

try:
    import cpi
except Exception:  # pragma: no cover
    cpi = None


st.set_page_config(page_title="Fin App", page_icon="💹", layout="wide")
st.title("Financial Data & Backtesting Suite")


def parse_periods(text: str) -> List[int]:
    if not text:
        return []
    try:
        return [int(x.strip()) for x in text.split(',') if x.strip()]
    except Exception:
        return []


def to_csv_download(df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=True).encode('utf-8')
    st.download_button("Download CSV", data=csv_bytes, file_name=filename, mime="text/csv")


def plot_price_and_returns(df: pd.DataFrame, window: int) -> None:
    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['adj_close'], mode='lines', name='Adj Close'))
    fig_price.update_layout(height=380, title="Adjusted Close")
    st.plotly_chart(fig_price, use_container_width=True)

    # Returns with bands & outliers
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(x=df.index, y=df['simple_rtn'], mode='lines', name='Simple Return'))
    if 'upper_band' in df.columns and 'lower_band' in df.columns:
        fig_ret.add_traces([
            go.Scatter(x=df.index, y=df['upper_band'], line=dict(color='orange', width=1), name='+3σ'),
            go.Scatter(x=df.index, y=df['lower_band'], line=dict(color='orange', width=1), name='-3σ', fill='tonexty', fillcolor='rgba(255,165,0,0.2)')
        ])
    if 'is_outlier' in df.columns:
        outs = df[df['is_outlier'] == 1]
        if not outs.empty:
            fig_ret.add_trace(go.Scatter(x=outs.index, y=outs['simple_rtn'], mode='markers', name='Outliers', marker=dict(color='red', size=6)))
    fig_ret.update_layout(height=380, title=f"Simple Returns with ±3σ Bands (window={window})")
    st.plotly_chart(fig_ret, use_container_width=True)


def make_candlestick(df: pd.DataFrame, smas: List[int], emas: List[int], rsi_period: int, macd: Optional[List[int]], bbands: Optional[List[float]], add_volume: bool):
    rows = 1
    row_heights = [0.7]
    specs = [[{"secondary_y": True}]]
    if rsi_period and rsi_period > 0:
        rows += 1
        row_heights.append(0.15)
        specs.append([{}])
    if macd and len(macd) == 3:
        rows += 1
        row_heights.append(0.15)
        specs.append([{}])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, specs=specs, vertical_spacing=0.03)

    # Main price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)

    # SMA/EMA overlays
    for p in smas:
        sma = df['Close'].rolling(window=p, min_periods=p).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma, mode='lines', name=f'SMA {p}'), row=1, col=1)
    for p in emas:
        ema = df['Close'].ewm(span=p, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema, mode='lines', name=f'EMA {p}'), row=1, col=1)

    # Bollinger Bands
    if bbands and len(bbands) >= 1 and bbands[0] > 0:
        period = int(bbands[0])
        std_mult = float(bbands[1]) if len(bbands) > 1 else 2.0
        mavg = df['Close'].rolling(window=period, min_periods=period).mean()
        mstd = df['Close'].rolling(window=period, min_periods=period).std(ddof=0)
        upper = mavg + std_mult * mstd
        lower = mavg - std_mult * mstd
        fig.add_trace(go.Scatter(x=df.index, y=upper, line=dict(color='gray', width=1), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, line=dict(color='gray', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.15)'), row=1, col=1)

    # Volume on secondary y-axis
    if add_volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(128,128,128,0.4)'), row=1, col=1, secondary_y=True)

    # RSI pane
    next_row = 2
    if rsi_period and rsi_period > 0:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f'RSI {rsi_period}', line=dict(color='purple')), row=next_row, col=1)
        fig.add_hline(y=70, line=dict(color='red', width=1, dash='dot'), row=next_row, col=1)
        fig.add_hline(y=30, line=dict(color='green', width=1, dash='dot'), row=next_row, col=1)
        next_row += 1

    # MACD pane
    if macd and len(macd) == 3:
        f, s, sig = macd
        ema_fast = df['Close'].ewm(span=f, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=s, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        hist = macd_line - signal_line
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name='MACD', line=dict(color='blue')), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal', line=dict(color='orange')), row=next_row, col=1)
        fig.add_trace(go.Bar(x=df.index, y=hist, name='Hist', marker_color='gray'), row=next_row, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


class SmaCross(bt.Strategy):
    params = dict(pfast=10, pslow=20)
    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.params.pfast)
        sma_slow = bt.ind.SMA(period=self.params.pslow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)
        self.trade_logs = []
        self.equity_logs = []
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()
        self.equity_logs.append(dict(dt=str(self.datas[0].datetime.date(0)), value=float(self.broker.getvalue())))
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_logs.append(dict(dt=str(self.datas[0].datetime.date(0)), pnl=float(trade.pnl), pnlcomm=float(trade.pnlcomm), price=float(trade.price), size=int(trade.size)))


tabs = st.tabs(["Data & Returns", "Candlestick", "Backtest", "Time Series & Forecasting", "Regression & ANOVA", "Asset Pricing", "About"])

with tabs[0]:
    st.subheader("Fetch data and analyze returns/outliers")
    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("Source", ["yahoo", "quandl", "intrinio"], index=0)
        ticker = st.text_input("Ticker", value="AAPL")
        start = st.date_input("Start", value=dt.date(2024,1,1))
        end = st.date_input("End", value=dt.date(2024,6,30))
        window = st.slider("Outlier window (days)", 5, 60, 21)
    with col2:
        dataset = st.text_input("Quandl dataset (e.g., EOD)")
        quandl_key = st.text_input("Quandl API key", type="password")
        intrinio_key = st.text_input("Intrinio API key", type="password")
        frequency = st.selectbox("Intrinio frequency", ["daily", "weekly", "monthly"], index=0)
    with col3:
        run_btn = st.button("Fetch & Analyze", use_container_width=True)

    if run_btn:
        try:
            if source == 'yahoo':
                prices = fetch_from_yahoo(ticker, str(start), str(end))
            elif source == 'quandl':
                prices = fetch_from_quandl(dataset, ticker, str(start), str(end), api_key=quandl_key or None)
            else:
                prices = fetch_from_intrinio(ticker, str(start), str(end), frequency=frequency, api_key=intrinio_key or None)
            df = compute_returns(prices)
            df = flag_outliers(df, window=window, z=3.0)
            st.success(f"Fetched {len(df)} rows for {ticker}")
            plot_price_and_returns(df, window)
            st.dataframe(df.tail(50))
            to_csv_download(df, f"{ticker}_analysis.csv")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[1]:
    st.subheader("Candlestick with indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        tk = st.text_input("Ticker", value="AAPL", key="cs_ticker")
        cs_start = st.date_input("Start", value=dt.date(2024,1,1), key="cs_start")
        cs_end = st.date_input("End", value=dt.date(2024,6,30), key="cs_end")
        add_volume = st.checkbox("Show volume", value=True)
    with col2:
        sma_text = st.text_input("SMAs (comma)", value="20,50")
        ema_text = st.text_input("EMAs (comma)", value="12,26")
        rsi_period = st.number_input("RSI period", min_value=0, max_value=200, value=14)
    with col3:
        macd_text = st.text_input("MACD fast,slow,signal", value="12,26,9")
        bb_text = st.text_input("BBands period,std", value="20,2")
        cs_btn = st.button("Render Chart", use_container_width=True)

    if cs_btn:
        try:
            ohlcv = yf.download(tk, start=str(cs_start), end=str(cs_end), auto_adjust=False, progress=False)
            if isinstance(ohlcv.columns, pd.MultiIndex):
                tickers = ohlcv.columns.get_level_values(1).unique().tolist()
                if len(tickers) == 1:
                    ohlcv = ohlcv.droplevel(1, axis=1)
                else:
                    if tk in tickers:
                        ohlcv = ohlcv.xs(tk, axis=1, level=1)
                    else:
                        ohlcv = ohlcv.xs(tickers[0], axis=1, level=1)
            ohlcv = ohlcv[['Open','High','Low','Close','Volume']].copy()
            smas = parse_periods(sma_text)
            emas = parse_periods(ema_text)
            macd = [int(x) for x in macd_text.split(',')] if macd_text and len(macd_text.split(',')) == 3 else None
            bbands = [float(x) for x in bb_text.split(',')] if bb_text else None
            make_candlestick(ohlcv, smas, emas, int(rsi_period), macd, bbands, add_volume)
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[2]:
    st.subheader("Backtest SMA Crossover")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_ticker = st.text_input("Ticker", value="AAPL", key="bt_tk")
        bt_start = st.date_input("Start", value=dt.date(2024,1,1), key="bt_start")
        bt_end = st.date_input("End", value=dt.date(2024,6,30), key="bt_end")
        bt_cash = st.number_input("Initial Cash", min_value=0.0, value=10000.0)
    with col2:
        pfast = st.number_input("Fast SMA", min_value=1, max_value=250, value=10)
        pslow = st.number_input("Slow SMA", min_value=2, max_value=400, value=20)
        rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=1.0, value=0.02, step=0.005, format="%.3f")
    with col3:
        run_bt = st.button("Run Backtest", use_container_width=True)

    if run_bt:
        try:
            df = yf.download(bt_ticker, start=str(bt_start), end=str(bt_end), auto_adjust=False, progress=False)
            if df is None or df.empty:
                st.warning("No data.")
            else:
                df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
                df.columns = ['open','high','low','close','adj_close','volume']
                df.dropna(inplace=True)

                data = bt.feeds.PandasData(dataname=df, open='open', high='high', low='low', close='close', volume='volume')
                cerebro = bt.Cerebro()
                cerebro.broker.setcash(bt_cash)
                cerebro.adddata(data)
                cerebro.addstrategy(SmaCross, pfast=int(pfast), pslow=int(pslow))
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=rf)
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
                results = cerebro.run(runonce=False)
                final_value = cerebro.broker.getvalue()

                strat = results[0]
                sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
                dd_info = strat.analyzers.dd.get_analysis()
                max_dd = None
                if isinstance(dd_info, dict):
                    max_dd = (dd_info.get('max') or {}).get('drawdown')
                num_days = max(1, (df.index[-1] - df.index[0]).days)
                cagr = (final_value / bt_cash) ** (365.0 / num_days) - 1.0

                st.metric("Final Portfolio Value", f"${final_value:,.2f}")
                st.write(f"CAGR: {cagr:.2%}")
                if sharpe is not None:
                    st.write(f"Sharpe (daily, rf={rf:.2%}): {sharpe:.3f}")
                if max_dd is not None:
                    st.write(f"Max Drawdown: {max_dd:.2f}%")

                # Equity curve
                eq = pd.DataFrame(getattr(strat, 'equity_logs', []))
                if not eq.empty:
                    eq['dt'] = pd.to_datetime(eq['dt'])
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=eq['dt'], y=eq['value'], name='Equity'))
                    fig_eq.update_layout(height=380, title="Equity Curve")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    to_csv_download(eq.set_index('dt'), f"{bt_ticker}_equity.csv")

                # Trades
                trades = pd.DataFrame(getattr(strat, 'trade_logs', []))
                if not trades.empty:
                    st.dataframe(trades.tail(50))
                    to_csv_download(trades, f"{bt_ticker}_trades.csv")

        except Exception as e:
            st.error(f"Error: {e}")

with tabs[3]:
    st.subheader("Time Series Analysis & Forecasting")
    st.markdown("Use the controls below to decompose, test stationarity, transform, and model a univariate time series.")

    # Controls
    colA, colB, colC = st.columns(3)
    with colA:
        ts_source = st.selectbox("Source", ["yahoo", "quandl"], index=0, key="tsa_src")
        ts_ticker = st.text_input("Ticker / Code", value="AAPL", key="tsa_tk")
        ts_dataset = st.text_input("Quandl dataset (e.g., EOD)", key="tsa_ds")
        ts_start = st.date_input("Start", value=dt.date(2018,1,1), key="tsa_start")
        ts_end = st.date_input("End", value=dt.date(2024,12,31), key="tsa_end")
    with colB:
        ts_freq = st.selectbox("Resample Frequency", ["D", "W", "M"], index=2)
        apply_log = st.checkbox("Log transform", value=False)
        apply_diff = st.checkbox("First difference", value=False)
        apply_cpi = st.checkbox("Deflate by US CPI (to end date)", value=False, help="Requires 'cpi' package and internet access")
    with colC:
        do_decomp = st.checkbox("Seasonal Decomposition", value=True)
        decomp_model = st.selectbox("Decomposition Model", ["additive", "multiplicative"], index=1)
        do_prophet = st.checkbox("Prophet Forecast", value=False)
        horizon_days = st.number_input("Prophet horizon (days)", min_value=1, max_value=1095, value=365)
    st.divider()

    colD, colE = st.columns(2)
    with colD:
        do_tests = st.checkbox("Stationarity Tests (ADF/KPSS)", value=True)
        acf_lags = st.slider("ACF/PACF Lags", 5, 120, 36)
    with colE:
        do_ses_holt = st.checkbox("Exponential Smoothing (SES/Holt)", value=False)
        holt_exp = st.checkbox("Use exponential trend (Holt)", value=True)
        holt_damped = st.checkbox("Use damped trend (Holt)", value=True)
        do_arima = st.checkbox("ARIMA (manual/auto)", value=False)
        use_auto = st.checkbox("Auto ARIMA (AIC)", value=True)
        p = st.number_input("p", 0, 5, 1)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 5, 1)

    run_tsa = st.button("Run Time Series Workflow", type="primary")

    def load_base_series() -> pd.Series:
        if ts_source == 'yahoo':
            dfp = fetch_from_yahoo(ts_ticker, str(ts_start), str(ts_end))
        else:
            dfp = fetch_from_quandl(ts_dataset, ts_ticker, str(ts_start), str(ts_end))
        s = dfp['adj_close'].astype(float).copy()
        s.name = 'price'
        return s

    if run_tsa:
        try:
            s = load_base_series()
            # Resample
            s_resampled = s.resample(ts_freq).last().dropna()

            # CPI deflation
            if apply_cpi:
                if cpi is None:
                    st.warning("cpi package not installed; skipping deflation.")
                else:
                    try:
                        target = f"{ts_end:%Y-%m}"
                        df_tmp = s_resampled.to_frame('value').reset_index().rename(columns={'index': 'date'})
                        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
                        def _adj(row):
                            try:
                                return cpi.inflate(row['value'], row['date'].strftime('%Y-%m'), target)
                            except Exception:
                                return np.nan
                        df_tmp['deflated'] = df_tmp.apply(_adj, axis=1)
                        s_resampled = df_tmp.set_index('date')['deflated'].astype(float).dropna()
                    except Exception as e:
                        st.warning(f"CPI deflation failed: {e}")

            # Log transform
            if apply_log:
                s_resampled = np.log(s_resampled.replace(0, np.nan)).dropna()

            # Differencing
            if apply_diff:
                s_resampled = s_resampled.diff().dropna()

            st.write(f"Series length after processing: {len(s_resampled)}")
            st.line_chart(s_resampled)

            # Stationarity tests
            if do_tests and len(s_resampled) > 12:
                adf_stat, adf_p, _, _, crit_vals, _ = adfuller(s_resampled.dropna(), autolag='AIC')
                st.write(f"ADF: stat={adf_stat:.4f}, p={adf_p:.4f}, crit={crit_vals}")
                try:
                    kpss_stat, kpss_p, _, kpss_crit = kpss(s_resampled.dropna(), regression='c', nlags='auto')
                    st.write(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f}, crit={kpss_crit}")
                except Exception as e:
                    st.write(f"KPSS failed: {e}")

                # ACF/PACF
                import matplotlib.pyplot as plt
                safe_lags = max(1, min(int(acf_lags), len(s_resampled) - 1))
                fig1, ax1 = plt.subplots(figsize=(6,3))
                plot_acf(s_resampled, ax=ax1, lags=safe_lags)
                st.pyplot(fig1)
                fig2, ax2 = plt.subplots(figsize=(6,3))
                plot_pacf(s_resampled, ax=ax2, lags=safe_lags, method='ywm')
                st.pyplot(fig2)

            # Decomposition
            if do_decomp and len(s_resampled) > 24:
                result = seasonal_decompose(s_resampled, model=decomp_model, period=None)
                import matplotlib.pyplot as plt
                fig = result.plot()
                st.pyplot(fig)

            # Prophet
            if do_prophet:
                if Prophet is None:
                    st.warning("prophet not installed.")
                else:
                    df_daily = s.resample('D').last().dropna().reset_index()
                    df_daily.columns = ['ds', 'y']
                    # Train/test split (last 10% as test)
                    split_idx = int(len(df_daily) * 0.9)
                    train = df_daily.iloc[:split_idx]
                    model = Prophet(seasonality_mode='additive')
                    model.fit(train)
                    future = model.make_future_dataframe(periods=int(horizon_days))
                    forecast = model.predict(future)
                    import matplotlib.pyplot as plt
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

            # SES / Holt
            if do_ses_holt and len(s_resampled) > 24:
                # Train/test split
                split_idx = int(len(s_resampled) * 0.8)
                train, test = s_resampled.iloc[:split_idx], s_resampled.iloc[split_idx:]
                # SES
                ses_fit = SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
                ses_pred = np.asarray(ses_fit.forecast(len(test)))
                ses_fc = pd.Series(ses_pred, index=test.index[: len(ses_pred)])
                # Holt
                trend_type = 'mul' if holt_exp else 'add'
                holt_model = ExponentialSmoothing(train, trend=trend_type, damped_trend=holt_damped, seasonal=None)
                holt_fit = holt_model.fit(optimized=True)
                holt_pred = np.asarray(holt_fit.forecast(len(test)))
                holt_fc = pd.Series(holt_pred, index=test.index[: len(holt_pred)])
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8,4))
                train.plot(ax=ax, label='Train')
                test.plot(ax=ax, label='Test')
                ses_fc.plot(ax=ax, label='SES Forecast')
                holt_fc.plot(ax=ax, label='Holt Forecast')
                ax.legend()
                st.pyplot(fig)

            # ARIMA
            if do_arima and len(s_resampled) > 36:
                split_idx = int(len(s_resampled) * 0.85)
                train, test = s_resampled.iloc[:split_idx], s_resampled.iloc[split_idx:]
                if use_auto and pm is not None:
                    model = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
                    order = model.order
                    st.write(f"Auto ARIMA selected order: {order}")
                    apred = np.asarray(model.predict(n_periods=len(test)))
                    fc = pd.Series(apred, index=test.index[: len(apred)])
                else:
                    order = (int(p), int(d), int(q))
                    arima_fit = ARIMA(train, order=order).fit()
                    st.text(arima_fit.summary())
                    mpred = np.asarray(arima_fit.forecast(steps=len(test)))
                    fc = pd.Series(mpred, index=test.index[: len(mpred)])
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8,4))
                train.plot(ax=ax, label='Train')
                test.plot(ax=ax, label='Test')
                fc.plot(ax=ax, label='ARIMA Forecast')
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

with tabs[4]:
    st.subheader("Regression & ANOVA")
    st.markdown("Run multiple regression on stock returns and perform ANOVA across selected stocks.")

    colR1, colR2, colR3 = st.columns(3)
    with colR1:
        reg_tickers_text = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOG")
        reg_target = st.text_input("Target (dependent) ticker", value="AAPL")
        reg_start = st.date_input("Start", value=dt.date(2023,1,1), key="reg_start")
        reg_end = st.date_input("End", value=dt.date(2024,12,31), key="reg_end")
    with colR2:
        reg_freq = st.selectbox("Resample", ["D","W","M"], index=0)
        reg_return_type = st.selectbox("Return type", ["simple","log"], index=0)
        reg_add_const = st.checkbox("Add intercept", value=True)
        reg_dropna = st.checkbox("Drop rows with NA across series", value=True)
    with colR3:
        run_reg = st.button("Run Regression & ANOVA", type="primary")

    def fetch_adj_close_series(ticker: str, start: str, end: str) -> pd.Series:
        dfp = fetch_from_yahoo(ticker, start, end)
        s = dfp['adj_close'].astype(float).copy()
        s.name = ticker
        return s

    def compute_returns_from_series(s: pd.Series, rtype: str) -> pd.Series:
        if rtype == 'simple':
            rtn = s.pct_change()
        else:
            rtn = np.log(s / s.shift(1))
        rtn.name = s.name
        return rtn

    if run_reg:
        try:
            tickers = [t.strip().upper() for t in reg_tickers_text.split(',') if t.strip()]
            if reg_target.upper() not in tickers:
                tickers = [reg_target.upper()] + tickers
            data_cols = {}
            for tk in tickers:
                s = fetch_adj_close_series(tk, str(reg_start), str(reg_end))
                s = s.resample(reg_freq).last().dropna()
                data_cols[tk] = s
            # Align all series on common dates
            df_prices = pd.concat(data_cols.values(), axis=1)
            df_prices.columns = list(data_cols.keys())
            # Returns
            df_rtn = df_prices.apply(lambda col: compute_returns_from_series(col, reg_return_type))
            if reg_dropna:
                df_rtn = df_rtn.dropna(how='any')
            else:
                df_rtn = df_rtn.dropna(how='all')

            st.write(f"Aligned returns shape: {df_rtn.shape}")
            st.dataframe(df_rtn.tail(20))

            y_name = reg_target.upper()
            if y_name not in df_rtn.columns:
                st.error("Target ticker not present in data after processing.")
            else:
                X_cols = [c for c in df_rtn.columns if c != y_name]
                if not X_cols:
                    st.warning("Need at least one predictor ticker besides the target.")
                else:
                    y = df_rtn[y_name]
                    X = df_rtn[X_cols]
                    if reg_add_const:
                        X = sm.add_constant(X, has_constant='add')
                    model = sm.OLS(y, X, missing='drop')
                    results = model.fit()

                    st.markdown("**Regression coefficients**")
                    coef_df = results.summary2().tables[1]
                    st.dataframe(coef_df)
                    st.markdown(
                        f"R²: {results.rsquared:.4f}  |  Adj. R²: {results.rsquared_adj:.4f}  |  F-stat: {results.fvalue:.3f} (p={results.f_pvalue:.4g})  |  n={int(results.nobs)}"
                    )

                    # Residual diagnostics
                    fitted = results.fittedvalues
                    resid = results.resid
                    fig_res = make_subplots(rows=1, cols=2, subplot_titles=("Residuals vs Fitted", "Residuals Histogram"))
                    fig_res.add_trace(go.Scatter(x=fitted, y=resid, mode='markers', name='resid'), row=1, col=1)
                    fig_res.add_hline(y=0, line=dict(color='gray', width=1), row=1, col=1)
                    fig_res.add_trace(go.Histogram(x=resid, nbinsx=30, name='hist'), row=1, col=2)
                    fig_res.update_layout(height=400)
                    st.plotly_chart(fig_res, use_container_width=True)

                    # One-way ANOVA across tickers (mean returns equal?)
                    df_long = df_rtn.reset_index().melt(id_vars=[df_rtn.index.name or 'index'], var_name='ticker', value_name='rtn')
                    df_long = df_long.dropna(subset=['rtn'])
                    if len(df_long) > 0:
                        mod = smf.ols('rtn ~ C(ticker)', data=df_long).fit()
                        anova_tbl = anova_lm(mod, typ=2)
                        st.markdown("**One-way ANOVA: rtn ~ C(ticker)**")
                        st.dataframe(anova_tbl)

                    # Download buttons
                    st.download_button(
                        "Download regression summary",
                        data=str(results.summary()).encode('utf-8'),
                        file_name=f"regression_{y_name}_on_{'_'.join(X_cols)}.txt",
                        mime="text/plain",
                    )
                    csv_buf = io.StringIO()
                    coef_df.to_csv(csv_buf)
                    st.download_button(
                        "Download coefficients CSV",
                        data=csv_buf.getvalue().encode('utf-8'),
                        file_name=f"coefficients_{y_name}.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error: {e}")

with tabs[5]:
    st.subheader("Asset Pricing Models: CAPM & Fama-French")
    colA, colB, colC = st.columns(3)
    with colA:
        ap_ticker = st.text_input("Asset Ticker", value="AMZN")
        ap_market = st.text_input("Market Benchmark", value="^GSPC", help="e.g., ^GSPC for S&P 500")
        ap_start = st.date_input("Start", value=dt.date(2015,1,1), key="ap_start")
        ap_end = st.date_input("End", value=dt.date(2025,1,1), key="ap_end")
        ap_freq = st.selectbox("Frequency", ["M"], index=0, help="Monthly frequency recommended for FF data")
    with colB:
        capm_rf_zero = st.checkbox("CAPM: Assume rf=0", value=True)
        do_capm = st.checkbox("Run CAPM", value=True)
        do_ff3 = st.checkbox("Run Fama-French 3-Factor", value=False)
        do_4f = st.checkbox("Run 4-Factor (with Momentum)", value=False)
        do_5f = st.checkbox("Run 5-Factor (RMW, CMA)", value=False)
    with colC:
        rolling_win = st.number_input("Rolling window (months) for FF", min_value=12, max_value=240, value=60)
        do_rolling = st.checkbox("Compute rolling FF betas", value=False)
        run_ap = st.button("Run Models", type="primary")

    def get_monthly_returns(ticker: str, start: str, end: str) -> pd.Series:
        dfp = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if dfp is None or dfp.empty:
            return pd.Series(dtype=float)
        if isinstance(dfp.columns, pd.MultiIndex):
            # drop ticker level if present
            tickers = dfp.columns.get_level_values(1).unique().tolist()
            if len(tickers) == 1:
                dfp = dfp.droplevel(1, axis=1)
            else:
                if ticker in tickers:
                    dfp = dfp.xs(ticker, axis=1, level=1)
                else:
                    dfp = dfp.xs(tickers[0], axis=1, level=1)
        prices = dfp.get('Adj Close') if 'Adj Close' in dfp.columns else dfp.get('Close')
        s = prices.resample('ME').last().pct_change().dropna()
        s.name = ticker
        return s.astype(float)

    def get_ff_factors_monthly(include_mom: bool=False, include_5: bool=False) -> pd.DataFrame:
        if pdr is None:
            raise RuntimeError("pandas_datareader not installed; cannot fetch Fama-French factors.")
        # Base 3-factor
        ff3 = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench')[0].copy()
        ff3.index = ff3.index.to_timestamp('M')
        ff3 = ff3.rename(columns={
            'Mkt-RF': 'mkt',
            'SMB': 'smb',
            'HML': 'hml',
            'RF': 'rf',
        }).astype(float) / 100.0
        out = ff3
        if include_mom:
            mom = pdr.DataReader('F-F_Momentum_Factor', 'famafrench')[0].copy()
            mom.index = mom.index.to_timestamp('M')
            # Momentum column may have varying whitespace; normalize
            mom.columns = [c.strip().lower() for c in mom.columns]
            if 'mom' not in mom.columns:
                # sometimes named 'Mom   '
                mom.rename(columns={mom.columns[0]: 'mom'}, inplace=True)
            mom = mom.astype(float) / 100.0
            out = out.join(mom[['mom']], how='inner')
        if include_5:
            ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0].copy()
            ff5.index = ff5.index.to_timestamp('M')
            ff5 = ff5.rename(columns={
                'Mkt-RF': 'mkt',
                'SMB': 'smb',
                'HML': 'hml',
                'RMW': 'rmw',
                'CMA': 'cma',
                'RF': 'rf',
            }).astype(float) / 100.0
            # keep only rmw,cma to add
            out = out.join(ff5[['rmw','cma']], how='inner')
        return out.dropna()

    def run_capm(asset: str, market: str, start: str, end: str, rf_zero: bool=True):
        r_a = get_monthly_returns(asset, start, end)
        r_m = get_monthly_returns(market, start, end)
        df = pd.concat([r_a, r_m], axis=1, join='inner').dropna()
        if df.empty:
            st.warning("No overlapping data for CAPM.")
            return
        y = df[asset]
        X = df[[market]]
        X.columns = ['market']
        if not rf_zero and pdr is not None:
            ff3 = get_ff_factors_monthly(include_mom=False, include_5=False)
            merged = df.join(ff3[['rf']], how='inner')
            y = merged[asset] - merged['rf']
            X['market'] = merged['market'] - merged['rf']
        if True:
            X = sm.add_constant(X, has_constant='add')
        res = sm.OLS(y, X, missing='drop').fit()
        st.text(res.summary())
        # Scatter with fit line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X['market'], y=y, mode='markers', name='returns'))
        xline = np.linspace(X['market'].min(), X['market'].max(), 100)
        yline = res.params['const'] + res.params['market'] * xline
        fig.add_trace(go.Scatter(x=xline, y=yline, mode='lines', name='fit'))
        fig.update_layout(height=380, title=f"CAPM: {asset} vs {market}")
        st.plotly_chart(fig, use_container_width=True)

    def run_ff_models(asset: str, start: str, end: str, do3: bool, do4: bool, do5: bool, rolling: bool, win: int):
        r_a = get_monthly_returns(asset, start, end)
        if r_a.empty:
            st.warning("No asset returns for FF models.")
            return
        need_mom = do4
        need_5 = do5
        ff = get_ff_factors_monthly(include_mom=need_mom, include_5=need_5)
        df = pd.concat([r_a, ff], axis=1, join='inner').dropna()
        df.rename(columns={asset: 'asset'}, inplace=True)
        df['excess_rtn'] = df['asset'] - df['rf']
        outputs = []
        if do3:
            m3 = smf.ols('excess_rtn ~ mkt + smb + hml', data=df).fit()
            st.markdown("**FF 3-Factor Summary**")
            st.text(m3.summary())
            outputs.append(('FF3', m3))
        if do4 and 'mom' in df.columns:
            m4 = smf.ols('excess_rtn ~ mkt + smb + hml + mom', data=df).fit()
            st.markdown("**FF 4-Factor (with Momentum) Summary**")
            st.text(m4.summary())
            outputs.append(('FF4', m4))
        if do5 and set(['rmw','cma']).issubset(df.columns):
            m5 = smf.ols('excess_rtn ~ mkt + smb + hml + rmw + cma', data=df).fit()
            st.markdown("**FF 5-Factor Summary**")
            st.text(m5.summary())
            outputs.append(('FF5', m5))

        # Rolling betas
        if rolling and (do3 or do4 or do5):
            factors = ['mkt','smb','hml']
            if do4 and 'mom' in df.columns:
                factors.append('mom')
            if do5 and set(['rmw','cma']).issubset(df.columns):
                factors.extend(['rmw','cma'])
            roll_rows = []
            for i in range(win, len(df)+1):
                sub = df.iloc[i-win:i]
                formula = 'excess_rtn ~ ' + ' + '.join(factors)
                r = smf.ols(formula=formula, data=sub).fit()
                row = {'date': df.index[i-1]}
                for name, val in r.params.items():
                    row[name] = val
                roll_rows.append(row)
            if roll_rows:
                roll_df = pd.DataFrame(roll_rows).set_index('date')
                # Plot betas over time
                beta_cols = [c for c in roll_df.columns if c != 'Intercept' and c != 'const']
                fig = go.Figure()
                for c in beta_cols:
                    fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df[c], name=c))
                fig.update_layout(height=420, title=f"Rolling betas ({win}-month window)")
                st.plotly_chart(fig, use_container_width=True)

    if run_ap:
        try:
            if do_capm:
                run_capm(ap_ticker.upper(), ap_market.upper(), str(ap_start), str(ap_end), rf_zero=capm_rf_zero)
            if do_ff3 or do_4f or do_5f:
                run_ff_models(ap_ticker.upper(), str(ap_start), str(ap_end), do_ff3, do_4f, do_5f, do_rolling, int(rolling_win))
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[6]:
    st.markdown("""
    - Data sources: Yahoo Finance (no key), Quandl and Intrinio (API keys optional here).
    - Returns are computed as simple and log; outliers flagged via rolling mean/std with 3σ.
    - Candlestick supports SMA/EMA, RSI, MACD, Bollinger Bands; volume overlay.
    - Backtest uses a simple SMA crossover strategy with performance stats and CSV exports.
    - Time Series: decomposition (statsmodels), Prophet forecasting, stationarity tests (ADF/KPSS + ACF/PACF), SES/Holt, manual/auto ARIMA.
    - Asset Pricing: CAPM, FF 3-/4-/5-factor models with rolling betas.
    """)


