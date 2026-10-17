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
    df = add_candlestick_features(df)   # <-- 加呢行, 放喺最開頭技術指標計算之前
    
    df["hsi_ret_1d"] = df["hsi_close"].pct_change()
    df["hsi_ret_5d"] = df["hsi_close"].pct_change(5)
    df["hsi_ret_10d"] = df["hsi_close"].pct_change(10)
    df["overnight_gap"] = df["hsi_open"] / df["hsi_close"].shift(1) - 1

    for m in ["nasdaq", "sp500", "dow", "dxy", "vix", "us10y"]:
        col = f"{m}_close"
        if col in df.columns:
            df[f"{m}_ret"] = df[col].pct_change()
        else:
            df[f"{m}_ret"] = 0.0

    df["vix_level"] = df["vix_close"] if "vix_close" in df.columns else 0.0

    df["atr14"] = _atr(df["hsi_high"], df["hsi_low"], df["hsi_close"], 14)
    df["atr_pct"] = df["atr14"] / df["hsi_close"]

    df["rsi14"] = _rsi(df["hsi_close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["hsi_close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"], df["bb_position"] = \
        _bollinger(df["hsi_close"])

    df["vol_ratio"] = df["hsi_volume"] / df["hsi_volume"].rolling(20).mean()

    # ---- 代理/輔助特徵: 先fillna, 避免因個別特徵數據覆蓋期短而拖累dropna刪走大量歷史 ----
    for col, default in [
        ("topweight_avg_ret", 0.0),
        ("topweight_volume_ratio", 1.0),
        ("ah_premium_change_proxy", 0.0),
        ("tech_relative_strength_proxy", 0.0),
        ("usdcnh_ret_proxy", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    df["target_direction"] = (df["hsi_close"].shift(-1) > df["hsi_close"]).astype(int)
    df["target_ret"] = df["hsi_close"].shift(-1) / df["hsi_close"] - 1

    # 只對核心技術指標(需要warm-up period)同target做dropna
    core_cols = [
        "hsi_ret_1d", "hsi_ret_5d", "hsi_ret_10d", "overnight_gap",
        "atr14", "atr_pct", "rsi14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_position", "vol_ratio",
        "target_direction", "target_ret"
    ]
    df.dropna(subset=core_cols, inplace=True)
    return df


FEATURE_COLUMNS = [
    "hsi_ret_1d", "hsi_ret_5d", "hsi_ret_10d", "overnight_gap",
    "nasdaq_ret", "sp500_ret", "dow_ret", "dxy_ret", "vix_ret", "us10y_ret",
    "vix_level", "atr_pct", "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "vol_ratio", "topweight_avg_ret",
    "topweight_volume_ratio", "ah_premium_change_proxy",
    "tech_relative_strength_proxy", "usdcnh_ret_proxy",
    "candlestick_sentiment_score",
    "pattern_doji", "pattern_hammer", "pattern_engulfing",
    "pattern_morning_star", "pattern_evening_star",
]
