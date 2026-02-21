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

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
    df = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

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

    # To track the buy/sell signals
    bought_stocks = set()  # Set of stocks bought
    sold_stocks = set()    # Set of stocks sold

    for i in range(lookback, len(monthly_prices)):

        if regime.iloc[i] is False:
            continue

        momentum = (
            monthly_prices.iloc[i-1] /
            monthly_prices.iloc[i-lookback]
        ) - 1

        top = momentum.sort_values(ascending=False).head(top_n).index
        current_buys = set(top)

        # Compare to previous buy list and determine which to buy and which to sell
        new_buys = current_buys - bought_stocks
        new_sells = bought_stocks - current_buys

        # Add new buys to the portfolio
        bought_stocks.update(new_buys)
        # Remove old sells from the portfolio
        sold_stocks.update(new_sells)

        # Update weights
        weights.loc[monthly_prices.index[i], top] = 1.0 / top_n

    weights = weights.shift(1).fillna(0)

    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * TRANSACTION_COST

    portfolio_returns = (weights * returns).sum(axis=1) - cost
    equity = (1 + portfolio_returns).cumprod()

    return equity, bought_stocks, sold_stocks

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
# PERFORMANCE
# ==============================

def calculate_stats(equity):
    if len(equity) < 12:
        return np.nan, np.nan

    years = len(equity) / 12
    cagr = equity.iloc[-1] ** (1 / years) - 1

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    return cagr, max_dd

# ==============================
# OOS SPLIT
# ==============================

def split_oos(equity):
    oos = equity[equity.index >= SPLIT_DATE]

    if len(oos) < 12:
        return np.nan, np.nan

    return calculate_stats(oos)

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
            equity, bought_stocks, sold_stocks = run_strategy(monthly_prices, regime, lookback, top_n)

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

            print(f"Completed → Lookback {lookback} | Top {top_n}")

            # Format the Telegram message to include buy/sell actions
            message = (f"Strategy run complete.\n"
                       f"Top N: {top_n}, Lookback: {lookback}\n"
                       f"CAGR: {oos_cagr:.2%}\n"
                       f"Buy signals: {', '.join(new_buys) if new_buys else 'None'}\n"
                       f"Sell signals: {', '.join(new_sells) if new_sells else 'None'}\n")
            telegram_alert(message)

    results_df = pd.DataFrame(results)
    print("\n===== PARAMETER STABILITY RESULTS =====\n")
    print(results_df.sort_values("OOS_CAGR", ascending=False))

    print("\nProfessional Stability Test Complete.")

if __name__ == "__main__":
    main()