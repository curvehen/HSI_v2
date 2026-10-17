import numpy as np
import pandas as pd

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _bollinger(series, period=20, n_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid
    position = (series - lower) / (upper - lower)
    return mid, upper, lower, width, position

def build_features(df):
    df = df.copy()

    df["hsi_ret_1d"] = df["hsi_close"].pct_change()
    df["hsi_ret_5d"] = df["hsi_close"].pct_change(5)
    df["hsi_ret_10d"] = df["hsi_close"].pct_change(10)
    df["overnight_gap"] = df["hsi_open"] / df["hsi_close"].shift(1) - 1

    for m in ["nasdaq", "sp500", "dow", "dxy", "vix", "us10y"]:
        col = f"{m}_close"
        df[f"{m}_ret"] = df[col].pct_change()

    df["vix_level"] = df["vix_close"]

    df["atr14"] = _atr(df["hsi_high"], df["hsi_low"], df["hsi_close"], 14)
    df["atr_pct"] = df["atr14"] / df["hsi_close"]

    df["rsi14"] = _rsi(df["hsi_close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["hsi_close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"], df["bb_position"] = _bollinger(df["hsi_close"])

    df["vol_ratio"] = df["hsi_volume"] / df["hsi_volume"].rolling(20).mean()

    df["topweight_avg_ret"] = df["topweight_avg_ret"].fillna(0)

    # Target: next day direction & next day return
    df["target_direction"] = (df["hsi_close"].shift(-1) > df["hsi_close"]).astype(int)
    df["target_ret"] = df["hsi_close"].shift(-1) / df["hsi_close"] - 1

    df.dropna(inplace=True)
    return df

FEATURE_COLUMNS = [
    "hsi_ret_1d", "hsi_ret_5d", "hsi_ret_10d", "overnight_gap",
    "nasdaq_ret", "sp500_ret", "dow_ret", "dxy_ret", "vix_ret", "us10y_ret",
    "vix_level", "atr_pct", "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "vol_ratio", "topweight_avg_ret"
]
