import talib
import pandas as pd
import numpy as np

# 精選對日線預測較有實證價值嘅形態(避免61種全加做噪音)
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
    """
    用df嘅hsi_open/high/low/close計算蠟燭形態訊號。
    每個形態輸出 -100/0/100, 正規化做 -1/0/1 方便模型使用。
    """
    df = df.copy()
    o, h, l, c = df["hsi_open"], df["hsi_high"], df["hsi_low"], df["hsi_close"]

    for name, func in PATTERN_FUNCTIONS.items():
        try:
            raw_signal = func(o, h, l, c)
            df[f"pattern_{name}"] = np.sign(raw_signal)  # 轉做 -1/0/1
        except Exception as e:
            print(f"警告: 計算形態{name}失敗 ({e}), 以0填充")
            df[f"pattern_{name}"] = 0

    # 額外: 綜合蠟燭情緒分數(所有形態訊號加總, 正代表偏多, 負代表偏空)
    pattern_cols = [f"pattern_{name}" for name in PATTERN_FUNCTIONS.keys()]
    df["candlestick_sentiment_score"] = df[pattern_cols].sum(axis=1)

    return df
