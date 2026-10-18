"""
輕量級命中率追蹤: 獨立csv log, 唔依賴performance_tracker.py嘅內部schema,
避免同predict.py現有嘅PREDICTION_HISTORY_PATH撞用途。
"""

import os
import csv
from datetime import datetime

LOOKBACK_WINDOW = 20          # 命中率計算嘅滾動窗口(最多幾多次近期預測)
MIN_RECORDS_FOR_DISPLAY = 5   # 少於呢個樣本數, 顯示"—", 唔展示命中率

LOG_PATH = "predictions/watchlist_hit_log.csv"
FIELDNAMES = ["run_date", "target", "ticker", "predicted_direction",
              "base_close", "resolved", "actual_direction", "correct"]


def _load_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_log(rows):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def resolve_pending_predictions(df, ticker):
    """
    用最新df(含target_close)嘅歷史價位, 回填之前未解析嘅預測結果。
    邏輯: 若log內某行run_date喺df index出現, 而且df入面有佢下一個交易日嘅收市價,
    就可以判斷實際方向, 同predicted_direction比較, 寫返correct/resolved。
    """
    rows = _load_log()
    if not rows:
        return

    dates = df.index.strftime("%Y-%m-%d").tolist()
    changed = False

    for row in rows:
        if row["ticker"] != ticker or row["resolved"] == "True":
            continue
        run_date = row["run_date"]
        if run_date not in dates:
            continue
        idx = dates.index(run_date)
        if idx + 1 >= len(dates):
            continue  # 下一個交易日仲未出現, 保持pending

        base_close = float(row["base_close"])
        next_close = float(df["target_close"].iloc[idx + 1])
        actual_direction = 1 if next_close > base_close else -1

        row["actual_direction"] = str(actual_direction)
        row["correct"] = str(actual_direction == int(row["predicted_direction"]))
        row["resolved"] = "True"
        changed = True

    if changed:
        _save_log(rows)


def log_new_prediction(target_label, ticker, run_date, predicted_direction, base_close):
    rows = _load_log()
    rows.append({
        "run_date": run_date,
        "target": target_label,
        "ticker": ticker,
        "predicted_direction": str(predicted_direction),
        "base_close": str(base_close),
        "resolved": "False",
        "actual_direction": "",
        "correct": "",
    })
    _save_log(rows)


def get_hit_rate(ticker):
    """回傳 (顯示用字串, 樣本數)。樣本唔夠就回傳 ('—', n)"""
    rows = _load_log()
    resolved = [r for r in rows if r["ticker"] == ticker and r["resolved"] == "True"]
    resolved = resolved[-LOOKBACK_WINDOW:]
    n = len(resolved)

    if n < MIN_RECORDS_FOR_DISPLAY:
        return "—", n

    correct_count = sum(1 for r in resolved if r["correct"] == "True")
    rate = correct_count / n
    return f"{rate*100:.1f}%", n
