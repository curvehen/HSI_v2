import yfinance as yf
import pandas as pd
from config import HSI_TICKER, TOP_WEIGHT_TICKERS, MACRO_TICKERS, START_DATE

def _dl(ticker, start):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df

def fetch_all(start=START_DATE):
    hsi = _dl(HSI_TICKER, start)[["Open", "High", "Low", "Close", "Volume"]].copy()
    hsi.columns = ["hsi_open", "hsi_high", "hsi_low", "hsi_close", "hsi_volume"]

    macro = {}
    for name, tk in MACRO_TICKERS.items():
        d = _dl(tk, start)["Close"]
        macro[name] = d

    macro_df = pd.DataFrame(macro)
    macro_df.columns = [f"{c}_close" for c in macro_df.columns]

    weight_rets = []
    for tk in TOP_WEIGHT_TICKERS:
        try:
            d = _dl(tk, start)["Close"].pct_change()
            weight_rets.append(d.rename(tk))
        except Exception:
            continue
    weight_df = pd.concat(weight_rets, axis=1)
    weight_df["topweight_avg_ret"] = weight_df.mean(axis=1)

    df = hsi.join(macro_df, how="inner")
    df = df.join(weight_df[["topweight_avg_ret"]], how="left")
    df.dropna(subset=["hsi_close"], inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df
