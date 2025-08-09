#!/usr/bin/env python3
import argparse
import datetime as dt

import backtrader as bt
import yfinance as yf
import pandas as pd


class SmaCross(bt.Strategy):
    params = dict(pfast=10, pslow=20)

    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.params.pfast)
        sma_slow = bt.ind.SMA(period=self.params.pslow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)
        self.equity_logs = []

    def next(self):
        if not self.position:  # no position
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()
        # log equity each bar
        self.equity_logs.append(dict(
            dt=str(self.datas[0].datetime.date(0)),
            value=float(self.broker.getvalue()),
        ))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log_trade(trade)

    def notify_order(self, order):
        pass

    def log_trade(self, trade):
        # store trade info for later export
        if not hasattr(self, 'trade_logs'):
            self.trade_logs = []
        self.trade_logs.append(dict(
            dt=str(self.datas[0].datetime.date(0)),
            pnl=float(trade.pnl),
            pnlcomm=float(trade.pnlcomm),
            price=float(trade.price),
            size=int(trade.size),
        ))


def main():
    p = argparse.ArgumentParser(description='Backtest a simple SMA crossover strategy (backtrader).')
    p.add_argument('--ticker', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--cash', type=float, default=10000.0)
    p.add_argument('--pfast', type=int, default=10)
    p.add_argument('--pslow', type=int, default=20)
    p.add_argument('--no-plot', action='store_true', help='Skip plotting (useful for headless runs)')
    p.add_argument('--trades-csv', help='Path to save executed trades as CSV')
    p.add_argument('--equity-csv', help='Path to save equity curve (date,value) as CSV')
    p.add_argument('--rf', type=float, default=0.0, help='Risk-free rate for Sharpe (annualized, e.g., 0.02)')
    args = p.parse_args()

    # Use yfinance to fetch and feed via PandasData to avoid Yahoo API issues
    df = yf.download(args.ticker, start=args.start, end=args.end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise SystemExit(f'No data for {args.ticker}')
    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
    df.columns = ['open','high','low','close','adj_close','volume']
    df.dropna(inplace=True)
    data = bt.feeds.PandasData(
        dataname=df,
        open='open', high='high', low='low', close='close', volume='volume',
        fromdate=dt.datetime.strptime(args.start, '%Y-%m-%d'),
        todate=dt.datetime.strptime(args.end, '%Y-%m-%d'),
    )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, pfast=args.pfast, pslow=args.pslow)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=args.rf)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturns', timeframe=bt.TimeFrame.Days)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    # Run in step-by-step mode to avoid once-optimization issues on short samples
    results = cerebro.run(runonce=False)
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Performance stats
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    dd_info = strat.analyzers.dd.get_analysis()
    max_drawdown = None
    if isinstance(dd_info, dict):
        max_section = dd_info.get('max') or {}
        max_drawdown = max_section.get('drawdown')
    # CAGR
    initial_value = args.cash
    final_value = cerebro.broker.getvalue()
    num_days = (df.index[-1] - df.index[0]).days or 1
    cagr = (final_value / initial_value) ** (365.0 / num_days) - 1.0
    print(f'CAGR: {cagr:.4%}')
    if sharpe is not None:
        print(f'Sharpe (daily, rf={args.rf:.2%}): {sharpe:.3f}')
    if max_drawdown is not None:
        print(f'Max Drawdown: {max_drawdown:.2f}%')

    # Save trades if requested
    if args.trades_csv and len(cerebro.strats) > 0:
        strat = cerebro.strats[0][0]
        logs = getattr(strat, 'trade_logs', [])
        if logs:
            import os
            import csv
            os.makedirs(os.path.dirname(os.path.abspath(args.trades_csv)), exist_ok=True)
            with open(args.trades_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['dt','pnl','pnlcomm','price','size'])
                writer.writeheader()
                writer.writerows(logs)
            print(f"Saved trades to {os.path.abspath(args.trades_csv)}")

    # Save equity curve
    if args.equity_csv and len(cerebro.strats) > 0:
        strat = cerebro.strats[0][0]
        eq = getattr(strat, 'equity_logs', [])
        if eq:
            import os
            import csv
            os.makedirs(os.path.dirname(os.path.abspath(args.equity_csv)), exist_ok=True)
            with open(args.equity_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['dt','value'])
                writer.writeheader()
                writer.writerows(eq)
            print(f"Saved equity curve to {os.path.abspath(args.equity_csv)}")

    if not args.no_plot:
        cerebro.plot(iplot=True, volume=False)


if __name__ == '__main__':
    main()
