import talib
import numpy as np

PATTERN_FUNCTIONS = {
    "doji": talib.CDLDOJI,
    "hammer": talib.CDLHAMMER,
    "hanging_man": talib.CDLHANGINGMAN,
    "engulfing": talib.CDLENGULFING,
    "morning_star": talib.CDLMORNINGSTAR,
    "evening_star": talib.CDLEVENINGSTAR,
    "shooting_star": talib.CDLSHOOTINGSTAR,
    "three_white_soldiers": talib.CDL3WHITESOLDIERS,
    "three_black_crows": talib.CDL3BLACKCROWS,
    "harami": talib.CDLHARAMI,
}


def add_candlestick_features(df):
    df = df.copy()
    o, h, l, c = df["target_open"], df["target_high"], df["target_low"], df["target_close"]

    for name, func in PATTERN_FUNCTIONS.items():
        try:
            raw_signal = func(o, h, l, c)
            df[f"pattern_{name}"] = np.sign(raw_signal)
        except Exception as e:
            print(f"警告: 計算形態{name}失敗 ({e}), 以0填充")
            df[f"pattern_{name}"] = 0

    pattern_cols = [f"pattern_{name}" for name in PATTERN_FUNCTIONS.keys()]
    df["candlestick_sentiment_score"] = df[pattern_cols].sum(axis=1)
    return df
