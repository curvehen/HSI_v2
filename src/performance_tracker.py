import pandas as pd
import os
from datetime import datetime
from config import PREDICTION_HISTORY_PATH, PERFORMANCE_LOG_PATH


def log_prediction(prediction_dict):
    """
    每次predict.py跑完, 將預測結果append落歷史CSV, 等之後可以同實際結果比較。
    """
    os.makedirs(os.path.dirname(PREDICTION_HISTORY_PATH), exist_ok=True)
    row = pd.DataFrame([prediction_dict])

    if os.path.exists(PREDICTION_HISTORY_PATH):
        existing = pd.read_csv(PREDICTION_HISTORY_PATH)
        existing = existing[existing["run_date"] != prediction_dict["run_date"]]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_csv(PREDICTION_HISTORY_PATH, index=False)
    print(f"預測已記錄落 {PREDICTION_HISTORY_PATH}")


def evaluate_pending_predictions(actual_df):
    """
    檢查prediction_history入面有冇"已經有實際結果但未評估"嘅紀錄,
    逐一計算誤差並寫入performance_log.csv。
    actual_df: 必須有 hsi_open/hsi_high/hsi_low/hsi_close, index=date
    """
    if not os.path.exists(PREDICTION_HISTORY_PATH):
        print("冇任何歷史預測紀錄, 跳過評估")
        return None

    history = pd.read_csv(PREDICTION_HISTORY_PATH, parse_dates=["run_date"])

    if os.path.exists(PERFORMANCE_LOG_PATH):
        evaluated = pd.read_csv(PERFORMANCE_LOG_PATH, parse_dates=["run_date"])
        evaluated_dates = set(evaluated["run_date"])
    else:
        evaluated = pd.DataFrame()
        evaluated_dates = set()

    new_records = []

    for _, row in history.iterrows():
        target_date = row["run_date"] + pd.Timedelta(days=1)
        # 找下一個實際有交易嘅日期(跳過週末/假期)
        future_dates = actual_df.index[actual_df.index > row["run_date"]]
        if len(future_dates) == 0:
            continue
        actual_date = future_dates[0]

        if row["run_date"] in evaluated_dates:
            continue

        actual_row = actual_df.loc[actual_date]
        actual_direction_up = actual_row["hsi_close"] > actual_row["hsi_open"]
        predicted_direction_up = row["ensemble_proba_up"] > 0.5

        direction_correct = int(predicted_direction_up == actual_direction_up)
        open_error_pct = abs(row["predicted_open"] - actual_row["hsi_open"]) / actual_row["hsi_open"]
        high_error_pct = abs(row["predicted_high"] - actual_row["hsi_high"]) / actual_row["hsi_high"]
        low_error_pct = abs(row["predicted_low"] - actual_row["hsi_low"]) / actual_row["hsi_low"]
        close_error_pct = abs(row["predicted_close"] - actual_row["hsi_close"]) / actual_row["hsi_close"]

        new_records.append({
            "run_date": row["run_date"],
            "evaluated_against_date": actual_date,
            "predicted_proba_up": row["ensemble_proba_up"],
            "agreement": row.get("agreement", None),
            "direction_correct": direction_correct,
            "open_error_pct": round(open_error_pct, 5),
            "high_error_pct": round(high_error_pct, 5),
            "low_error_pct": round(low_error_pct, 5),
            "close_error_pct": round(close_error_pct, 5),
        })

    if len(new_records) == 0:
        print("冇新嘅已完成預測可供評估")
        return evaluated if len(evaluated) > 0 else None

    new_df = pd.DataFrame(new_records)
    combined = pd.concat([evaluated, new_df], ignore_index=True) if len(evaluated) > 0 else new_df
    combined.to_csv(PERFORMANCE_LOG_PATH, index=False)
    print(f"新增 {len(new_records)} 筆評估紀錄落 {PERFORMANCE_LOG_PATH}")
    return combined
