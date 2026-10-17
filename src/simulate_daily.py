import numpy as np
import pandas as pd
from src.models import (train_logistic, train_xgb, train_lgb,
                          predict_proba_logistic, predict_proba_tree)
from src.lstm_model import train_lstm, predict_lstm, SEQ_LEN
from src.ensemble import stack_predictions, agreement_score
from src.garch_model import fit_garch_forecast
from src.risk import predict_next_ohlc, compute_entry_exit
from src.features import FEATURE_COLUMNS
from config import SIM_START_DATE, SIM_END_DATE, LSTM_RETRAIN_FREQ_DAYS, MIN_TRAIN_HISTORY


def run_daily_simulation(df):
    df = df.sort_index()

    sim_start_ts = pd.Timestamp(SIM_START_DATE)
    valid_dates = df.index[df.index >= sim_start_ts]
    if len(valid_dates) == 0:
        raise ValueError(f"數據中冇任何日期 >= {SIM_START_DATE}")
    sim_start_idx = df.index.get_loc(valid_dates[0])

    if SIM_END_DATE is None:
        end_idx = len(df) - 1
    else:
        sim_end_ts = pd.Timestamp(SIM_END_DATE)
        valid_end_dates = df.index[df.index <= sim_end_ts]
        if len(valid_end_dates) == 0:
            raise ValueError(f"數據中冇任何日期 <= {SIM_END_DATE}")
        end_idx = df.index.get_loc(valid_end_dates[-1])

    if sim_start_idx < MIN_TRAIN_HISTORY:
        raise ValueError(
            f"訓練歷史不足{MIN_TRAIN_HISTORY}日(現有{sim_start_idx}日), 無法開始模擬"
        )

    results = []
    lstm_model = None
    days_since_lstm_retrain = 999
    skipped_days = 0

    for t_idx in range(sim_start_idx, end_idx + 1):
        train_slice = df.iloc[:t_idx]
        current_row = df.iloc[t_idx]

        train_slice_clean = train_slice.dropna(subset=["target_direction"])
        X_train = train_slice_clean[FEATURE_COLUMNS].values
        y_train = train_slice_clean["target_direction"].values

        try:
            logreg, scaler = train_logistic(X_train, y_train)
            xgb_model = train_xgb(X_train, y_train)
            lgb_model = train_lgb(X_train, y_train)
        except ValueError as e:
            print(f"跳過 {df.index[t_idx].date()}: 訓練失敗 ({e})")
            skipped_days += 1
            continue

        proba_train = {
            "logreg": predict_proba_logistic(logreg, scaler, X_train),
            "xgb": predict_proba_tree(xgb_model, X_train),
            "lgb": predict_proba_tree(lgb_model, X_train),
        }
        meta_model = stack_predictions(proba_train, y_train=y_train, train_mode=True)

        full_X_train_raw = train_slice[FEATURE_COLUMNS].values
        latest_X = full_X_train_raw[-1:].copy()
        proba_input = {
            "logreg": predict_proba_logistic(logreg, scaler, latest_X),
            "xgb": predict_proba_tree(xgb_model, latest_X),
            "lgb": predict_proba_tree(lgb_model, latest_X),
        }
        ensemble_proba = stack_predictions(proba_input, meta_model=meta_model)[0]

        if days_since_lstm_retrain >= LSTM_RETRAIN_FREQ_DAYS:
            lstm_model = train_lstm(X_train, y_train, n_features=len(FEATURE_COLUMNS))
            days_since_lstm_retrain = 0
        else:
            days_since_lstm_retrain += 1

        lstm_proba = None
        if lstm_model is not None and len(full_X_train_raw) >= SEQ_LEN:
            seq = full_X_train_raw[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))
            lstm_proba = predict_lstm(lstm_model, seq)

        agree = agreement_score(ensemble_proba, lstm_proba)
        garch_vol = fit_garch_forecast(train_slice["target_ret_1d"])

        last_close = train_slice["target_close"].iloc[-1]
        atr = train_slice["atr14"].iloc[-1]
        overnight_gap_est = train_slice["overnight_gap"].rolling(10).mean().iloc[-1]
        ohlc_pred = predict_next_ohlc(last_close, overnight_gap_est, atr, ensemble_proba)

        direction = 1 if ensemble_proba > 0.5 else -1
        entry, stop, target = compute_entry_exit(
            ohlc_pred["expected_open"], atr, 1 if direction == 1 else 0
        )

        actual_ret = current_row["target_ret_1d"]
        correct_direction = int((ensemble_proba > 0.5) == (actual_ret > 0))

        results.append({
            "date": df.index[t_idx],
            "predicted_proba_up": ensemble_proba,
            "lstm_proba_up": lstm_proba,
            "agreement": agree,
            "garch_vol_forecast": garch_vol,
            "predicted_open": ohlc_pred["expected_open"],
            "predicted_high": ohlc_pred["expected_high"],
            "predicted_low": ohlc_pred["expected_low"],
            "predicted_close": ohlc_pred["expected_close"],
            "suggested_entry": entry,
            "suggested_stop": stop,
            "suggested_target": target,
            "actual_open": current_row["target_open"],
            "actual_high": current_row["target_high"],
            "actual_low": current_row["target_low"],
            "actual_close": current_row["target_close"],
            "actual_ret": actual_ret,
            "correct_direction": correct_direction,
        })

        print(f"模擬: {df.index[t_idx].date()} | proba_up={ensemble_proba:.3f} | "
              f"方向{'啱' if correct_direction else '錯'}")

    if skipped_days > 0:
        print(f"注意: 共有 {skipped_days} 日因訓練失敗被跳過")

    if len(results) == 0:
        raise RuntimeError("模擬未產生任何結果")

    return pd.DataFrame(results).set_index("date")
