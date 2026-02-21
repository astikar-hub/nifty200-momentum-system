import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os
import requests

# ==============================
# SETTINGS
# ==============================

START_DATE = "2012-01-01"
SPLIT_DATE = "2019-01-01"
END_DATE = datetime.datetime.today().strftime("%Y-%m-%d")

BENCHMARK = "^NSEI"
TRANSACTION_COST = 0.002

LOOKBACK_LIST = [6, 9, 12, 15]
TOPN_LIST = [10, 15, 20]

DATA_CACHE_FILE = "nifty200_price_data.csv"

# ==============================
# LOAD SYMBOLS
# ==============================

def load_symbols():
    df = pd.read_csv("nifty200_symbols.csv")
    return (df["Symbol"].str.strip() + ".NS").tolist()

# ==============================
# DOWNLOAD / LOAD DATA
# ==============================

def get_data(symbols):

    if os.path.exists(DATA_CACHE_FILE):
        print("Loading cached price data...")
        return pd.read_csv(DATA_CACHE_FILE, index_col=0, parse_dates=True)

    print("Downloading tickers in batch...")
    data = yf.download(
        tickers=symbols,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True
    )

    prices = {}

    for sym in symbols:
        try:
            prices[sym] = data[sym]["Close"]
        except:
            continue

    prices = pd.DataFrame(prices)
    prices.to_csv(DATA_CACHE_FILE)

    return prices

# ==============================
# BENCHMARK REGIME (FIXED)
# ==============================

def get_regime():

    df = yf.download(BENCHMARK,
                     start=START_DATE,
                     end=END_DATE,
                     auto_adjust=True,
                     progress=False)

    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = df["Close"].dropna()

    dma200 = price.rolling(200).mean()
    regime = (price > dma200)

    monthly_regime = regime.resample("ME").last()

    # Ensure boolean series
    monthly_regime = monthly_regime.astype(bool)

    return monthly_regime

# ==============================
# STRATEGY ENGINE
# ==============================

def run_strategy(monthly_prices, regime, lookback, top_n):

    returns = monthly_prices.pct_change()

    weights = pd.DataFrame(
        0.0,
        index=monthly_prices.index,
        columns=monthly_prices.columns
    )

    # Align regime index
    regime = regime.reindex(monthly_prices.index).fillna(False)

    # Store signals for buy and sell
    buy_signals = []
    sell_signals = []

    for i in range(lookback, len(monthly_prices)):

        if regime.iloc[i] is False:
            continue

        momentum = (
            monthly_prices.iloc[i-1] /
            monthly_prices.iloc[i-lookback]
        ) - 1

        top = momentum.sort_values(ascending=False).head(top_n).index
        previous_top = weights.loc[monthly_prices.index[i-1]].nlargest(top_n).index

        # Identify buy/sell signals
        new_buys = list(set(top) - set(previous_top))
        sells = list(set(previous_top) - set(top))

        if new_buys:
            buy_signals.append((monthly_prices.index[i], new_buys))
        if sells:
            sell_signals.append((monthly_prices.index[i], sells))

        weights.loc[monthly_prices.index[i], top] = 1.0 / top_n

    weights = weights.shift(1).fillna(0)

    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * TRANSACTION_COST

    portfolio_returns = (weights * returns).sum(axis=1) - cost
    equity = (1 + portfolio_returns).cumprod()

    return equity, buy_signals, sell_signals

# ==============================
# TELEGRAM ALERT
# ==============================

def telegram_alert(message: str):

    if TELEGRAM_TOKEN is None or CHAT_ID is None:
        print("Telegram not configured")
        return

    url = (f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
           f"/sendMessage?chat_id={CHAT_ID}&text={message}")
    requests.get(url)

# ==============================
# MAIN
# ==============================

def main():

    print("Loading symbols...")
    symbols = load_symbols()

    prices = get_data(symbols)

    regime = get_regime()

    monthly_prices = prices.resample("ME").last()

    results = []

    print("\nRunning Stability Grid...\n")

    for lookback in LOOKBACK_LIST:
        for top_n in TOPN_LIST:

            equity, buy_signals, sell_signals = run_strategy(monthly_prices,
                                                             regime,
                                                             lookback,
                                                             top_n)

            full_cagr, full_dd = calculate_stats(equity)
            oos_cagr, oos_dd = split_oos(equity)

            results.append({
                "Lookback": lookback,
                "TopN": top_n,
                "Full_CAGR": full_cagr,
                "Full_MaxDD": full_dd,
                "OOS_CAGR": oos_cagr,
                "OOS_MaxDD": oos_dd
            })

            # Prepare Telegram message
            buy_msg = "\n".join([f"Date: {date}, Buy: {', '.join(stocks)}" for date, stocks in buy_signals])
            sell_msg = "\n".join([f"Date: {date}, Sell: {', '.join(stocks)}" for date, stocks in sell_signals])

            msg = f"Strategy run complete.\nTop N: {top_n}, Lookback: {lookback}\nCAGR: {oos_cagr:.2%}\nBuy Signals:\n{buy_msg}\nSell Signals:\n{sell_msg}"

            # Send the message
            telegram_alert(msg)

            print(f"Completed → Lookback {lookback} | Top {top_n}")

    results_df = pd.DataFrame(results)

    print("\n===== PARAMETER STABILITY RESULTS =====\n")
    print(results_df.sort_values("OOS_CAGR", ascending=False))

    print("\nProfessional Stability Test Complete.")

if __name__ == "__main__":
    main()