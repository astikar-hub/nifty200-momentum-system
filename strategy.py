import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os
import requests

# =====================================================
# SETTINGS
# =====================================================

START_DATE = "2012-01-01"
SPLIT_DATE = "2019-01-01"
END_DATE = datetime.datetime.today().strftime("%Y-%m-%d")

BENCHMARK = "^NSEI"
TRANSACTION_COST = 0.002

LOOKBACK_LIST = [6, 9, 12, 15]
TOPN_LIST = [10, 15, 20]

SYMBOL_FILE = "nifty200_symbols.csv"

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =====================================================
# LOAD SYMBOLS
# =====================================================

def load_symbols():
    df = pd.read_csv(SYMBOL_FILE)
    return (df["Symbol"].str.strip() + ".NS").tolist()

# =====================================================
# DOWNLOAD DATA (NO CACHE - CLOUD SAFE)
# =====================================================

def get_data(symbols):

    print("Downloading tickers in batch...")

    data = yf.download(
        tickers=symbols,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False
    )

    prices = {}

    for sym in symbols:
        try:
            prices[sym] = data[sym]["Close"]
        except:
            continue

    return pd.DataFrame(prices)

# =====================================================
# BENCHMARK REGIME FILTER
# =====================================================

def get_regime():

    df = yf.download(
        BENCHMARK,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = df["Close"].dropna()

    dma200 = price.rolling(200).mean()
    regime = price > dma200

    monthly_regime = regime.resample("ME").last().astype(bool)

    return monthly_regime

# =====================================================
# STRATEGY ENGINE
# =====================================================

def run_strategy(monthly_prices, regime, lookback, top_n):

    returns = monthly_prices.pct_change()

    weights = pd.DataFrame(
        0.0,
        index=monthly_prices.index,
        columns=monthly_prices.columns
    )

    regime = regime.reindex(monthly_prices.index).fillna(False)

    for i in range(lookback, len(monthly_prices)):

        if not regime.iloc[i]:
            continue

        momentum = (
            monthly_prices.iloc[i-1] /
            monthly_prices.iloc[i-lookback]
        ) - 1

        top = momentum.sort_values(ascending=False).head(top_n).index

        weights.loc[monthly_prices.index[i], top] = 1.0 / top_n

    weights = weights.shift(1).fillna(0)

    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * TRANSACTION_COST

    portfolio_returns = (weights * returns).sum(axis=1) - cost
    equity = (1 + portfolio_returns).cumprod()

    return equity

# =====================================================
# PERFORMANCE METRICS
# =====================================================

def calculate_stats(equity):

    if len(equity) < 12:
        return np.nan, np.nan

    years = len(equity) / 12
    cagr = equity.iloc[-1] ** (1 / years) - 1

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    return cagr, max_dd

def split_oos(equity):

    oos = equity[equity.index >= SPLIT_DATE]

    if len(oos) < 12:
        return np.nan, np.nan

    return calculate_stats(oos)

# =====================================================
# TELEGRAM ALERT
# =====================================================

def telegram_alert(message: str):

    if TELEGRAM_TOKEN is None or TELEGRAM_CHAT_ID is None:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload)
    except:
        pass

# =====================================================
# MAIN
# =====================================================

def main():

    try:

        print("Loading symbols...")
        symbols = load_symbols()

        prices = get_data(symbols)
        regime = get_regime()

        monthly_prices = prices.resample("ME").last()

        results = []

        print("\nRunning Stability Grid...\n")

        for lookback in LOOKBACK_LIST:
            for top_n in TOPN_LIST:

                equity = run_strategy(
                    monthly_prices,
                    regime,
                    lookback,
                    top_n
                )

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

        results_df = pd.DataFrame(results)

        best = results_df.sort_values(
            "OOS_CAGR",
            ascending=False
        ).iloc[0]

        print("\n===== PARAMETER STABILITY RESULTS =====\n")
        print(results_df.sort_values("OOS_CAGR", ascending=False))

        summary = (
            f"Strategy Run Complete\n"
            f"Best Parameter:\n"
            f"Lookback: {int(best['Lookback'])}\n"
            f"TopN: {int(best['TopN'])}\n"
            f"OOS CAGR: {best['OOS_CAGR']:.2%}\n"
            f"OOS MaxDD: {best['OOS_MaxDD']:.2%}"
        )

        print("\n" + summary)

        telegram_alert(summary)

    except Exception as e:

        error_msg = f"Strategy FAILED\nError: {str(e)}"
        print(error_msg)
        telegram_alert(error_msg)
        raise

if __name__ == "__main__":
    main()