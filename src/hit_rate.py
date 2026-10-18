import os
import pandas as pd
from datetime import datetime

HISTORY_PATH = "predictions/prediction_history.csv"
MIN_RECORDS_FOR_DISPLAY = 5
LOOKBACK_WINDOW = 20


def _load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=[
            "date", "target", "ticker", "proba_up",
            "predicted_direction", "last_known_close",
            "actual_close", "actual_direction", "correct"
        ])
    return pd.read_csv(HISTORY_PATH, parse_dates=["date"])


def _save_history(df):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)


def resolve_pending_predictions(current_price_lookup):
    """
    喺新一輪預測開始前調用。
    current_price_lookup: dict {ticker: 最新實際收市價}
    將上一次未回填嘅預測, 用而家嘅實際價格計算方向是否答啱。
    """
    df = _load_history()
    if df.empty:
        return df

    pending_mask = df["actual_close"].isna()
    for idx in df[pending_mask].index:
        ticker = df.at[idx, "ticker"]
        if ticker not in current_price_lookup:
            continue
        actual_close = current_price_lookup[ticker]
        last_close = df.at[idx, "last_known_close"]

        actual_direction = "up" if actual_close > last_close else "down"
        predicted_direction = df.at[idx, "predicted_direction"]
        correct = (actual_direction == predicted_direction)

        df.at[idx, "actual_close"] = actual_close
        df.at[idx, "actual_direction"] = actual_direction
        df.at[idx, "correct"] = correct

    _save_history(df)
    return df


def log_new_prediction(target, ticker, proba_up, last_known_close, run_date=None):
    """預測完成後調用, 存低今次嘅預測方向(actual_close暫留空, 等下次回填)"""
    df = _load_history()
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")

    predicted_direction = "up" if proba_up >= 0.5 else "down"

    new_row = {
        "date": run_date,
        "target": target,
        "ticker": ticker,
        "proba_up": proba_up,
        "predicted_direction": predicted_direction,
        "last_known_close": last_known_close,
        "actual_close": None,
        "actual_direction": None,
        "correct": None,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_history(df)


def get_hit_rate(ticker, lookback=LOOKBACK_WINDOW):
    """
    回傳(命中率字串, 樣本數)。
    樣本不足MIN_RECORDS_FOR_DISPLAY時, 回傳"—", 避免虛構數字。
    """
    df = _load_history()
    if df.empty:
        return "—", 0

    subset = df[(df["ticker"] == ticker) & (df["correct"].notna())]
    subset = subset.sort_values("date", ascending=False).head(lookback)

    n = len(subset)
    if n < MIN_RECORDS_FOR_DISPLAY:
        return "—", n

    hit_rate = subset["correct"].mean()
    return f"{hit_rate * 100:.1f}%", n
