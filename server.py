import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf

from fetch_and_analyze import (
    fetch_from_yahoo,
    fetch_from_quandl,
    compute_returns,
    flag_outliers,
)
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

app = FastAPI(title="Fin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_monthly_returns(ticker: str, start: str, end: str) -> pd.Series:
    dfp = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if dfp is None or dfp.empty:
        return pd.Series(dtype=float)
    if isinstance(dfp.columns, pd.MultiIndex):
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/returns")
def get_returns(source: str, ticker: str, start: str, end: str, window: int = 21):
    if source not in {"yahoo", "quandl"}:
        raise HTTPException(status_code=400, detail="source must be yahoo or quandl")
    if source == 'yahoo':
        prices = fetch_from_yahoo(ticker, start, end)
    else:
        # default to EOD if not given via query in future
        prices = fetch_from_quandl('EOD', ticker, start, end)
    df = compute_returns(prices)
    df = flag_outliers(df, window=window, z=3.0)
    df_out = df.reset_index().rename(columns={'index': 'date', df.index.name or 'index': 'date'})
    df_out['date'] = pd.to_datetime(df_out['date']).dt.strftime('%Y-%m-%d')
    return df_out.to_dict(orient='records')


@app.get("/capm")
def capm(asset: str, market: str, start: str, end: str, rf_zero: bool = True):
    r_a = get_monthly_returns(asset, start, end)
    r_m = get_monthly_returns(market, start, end)
    df = pd.concat([r_a, r_m], axis=1, join='inner').dropna()
    if df.empty:
        raise HTTPException(status_code=400, detail="No overlapping data")
    y = df[asset]
    X = df[[market]].rename(columns={market: 'market'})
    if not rf_zero and pdr is not None:
        ff3 = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench')[0].copy()
        ff3.index = ff3.index.to_timestamp('M')
        ff3 = ff3.rename(columns={'Mkt-RF': 'mkt', 'RF': 'rf'}).astype(float) / 100.0
        merged = df.join(ff3[['rf']], how='inner')
        y = merged[asset] - merged['rf']
        X['market'] = merged['market'] - merged['rf']
    X = sm.add_constant(X, has_constant='add')
    res = sm.OLS(y, X, missing='drop').fit()
    return {
        'alpha': float(res.params.get('const', np.nan)),
        'beta': float(res.params.get('market', np.nan)),
        'r2': float(res.rsquared),
        'n': int(res.nobs),
        'summary': str(res.summary()),
    }


@app.get("/ff3")
def ff3(asset: str, start: str, end: str):
    if pdr is None:
        raise HTTPException(status_code=400, detail='pandas_datareader not installed on server')
    r_a = get_monthly_returns(asset, start, end)
    ff = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench')[0].copy()
    ff.index = ff.index.to_timestamp('M')
    ff = ff.rename(columns={'Mkt-RF': 'mkt', 'SMB': 'smb', 'HML': 'hml', 'RF': 'rf'}).astype(float) / 100.0
    df = pd.concat([r_a.rename('asset'), ff], axis=1, join='inner').dropna()
    df['excess_rtn'] = df['asset'] - df['rf']
    res = smf.ols('excess_rtn ~ mkt + smb + hml', data=df).fit()
    return {
        'params': {k: float(v) for k, v in res.params.items()},
        'r2': float(res.rsquared),
        'n': int(res.nobs),
        'summary': str(res.summary()),
    }


