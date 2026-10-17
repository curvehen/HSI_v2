import numpy as np
import pandas as pd
from src.candlestick_patterns import add_candlestick_features


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
    return macd_line, signal_line, macd_line - signal_line


def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
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

    # ---- 蠟燭形態偵測 (用target_open/high/low/close, 放喺最前) ----
    df = add_candlestick_features(df)

    df["target_ret_1d"] = df["target_close"].pct_change()
    df["target_ret_5d"] = df["target_close"].pct_change(5)
    df["target_ret_10d"] = df["target_close"].pct_change(10)
    df["overnight_gap"] = df["target_open"] / df["target_close"].shift(1) - 1

    df["benchmark_ret"] = df["benchmark_ret"].fillna(0) if "benchmark_ret" in df.columns else 0.0

    for m in ["nasdaq", "sp500", "dow", "dxy", "vix", "us10y"]:
        col = f"{m}_close"
        df[f"{m}_ret"] = df[col].pct_change() if col in df.columns else 0.0

    df["vix_level"] = df["vix_close"] if "vix_close" in df.columns else 0.0

    df["atr14"] = _atr(df["target_high"], df["target_low"], df["target_close"], 14)
    df["atr_pct"] = df["atr14"] / df["target_close"]

    df["rsi14"] = _rsi(df["target_close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["target_close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"], df["bb_position"] = \
        _bollinger(df["target_close"])

    df["vol_ratio"] = df["target_volume"] / df["target_volume"].rolling(20).mean()

    for col, default in [
        ("topweight_avg_ret", 0.0),
        ("topweight_volume_ratio", 1.0),
        ("ah_premium_change_proxy", 0.0),
        ("tech_relative_strength_proxy", 0.0),
        ("usdcnh_ret_proxy", 0.0),
    ]:
        df[col] = df[col].fillna(default) if col in df.columns else default

    df["target_direction"] = (df["target_close"].shift(-1) > df["target_close"]).astype(int)
    df["target_ret"] = df["target_close"].shift(-1) / df["target_close"] - 1

    core_cols = [
        "target_ret_1d", "target_ret_5d", "target_ret_10d", "overnight_gap",
        "atr14", "atr_pct", "rsi14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_position", "vol_ratio", "target_direction", "target_ret"
    ]
    df.dropna(subset=core_cols, inplace=True)
    return df


FEATURE_COLUMNS = [
    "target_ret_1d", "target_ret_5d", "target_ret_10d", "overnight_gap",
    "benchmark_ret",
    "nasdaq_ret", "sp500_ret", "dow_ret", "dxy_ret", "vix_ret", "us10y_ret",
    "vix_level", "atr_pct", "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "vol_ratio", "topweight_avg_ret",
    "topweight_volume_ratio", "ah_premium_change_proxy",
    "tech_relative_strength_proxy", "usdcnh_ret_proxy",
    "candlestick_sentiment_score",
    "pattern_doji", "pattern_hammer", "pattern_engulfing",
    "pattern_morning_star", "pattern_evening_star",
]
