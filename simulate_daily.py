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
    """
    df: 已經build_features()完成嘅完整歷史dataframe (index=date, 已排序)
    邏輯: 對於SIM_START_DATE之後嘅每一個交易日t,
          只用 df.iloc[:t_idx] (即截至t-1收市) 嘅數據訓練+預測
    """
    df = df.sort_index()
    sim_start_idx = df.index.get_indexer([pd.Timestamp(SIM_START_DATE)], method="bfill")[0]
    end_idx = len(df) - 1 if SIM_END_DATE is None else \
        df.index.get_indexer([pd.Timestamp(SIM_END_DATE)], method="ffill")[0]

    if sim_start_idx < MIN_TRAIN_HISTORY:
        raise ValueError(f"訓練歷史不足{MIN_TRAIN_HISTORY}日, 無法喺{SIM_START_DATE}開始模擬")

    results = []
    lstm_model = None
    days_since_lstm_retrain = 999  # 強制第一次就訓練

    for t_idx in range(sim_start_idx, end_idx + 1):
        # ---- 關鍵: 只用截至 t-1 嘅數據, 完全睇唔到t當日或之後 ----
        train_slice = df.iloc[:t_idx]          # 訓練集: 截至t-1收市
        current_row = df.iloc[t_idx]            # t當日嘅真實數據(用嚟做驗證, 唔用嚟訓練)

        X_train = train_slice[FEATURE_COLUMNS].values
        y_train = train_slice["target_direction"].values

        # ---- 樹模型: 每日重訓 ----
        logreg, scaler = train_logistic(X_train, y_train)
        xgb_model = train_xgb(X_train, y_train)
        lgb_model = train_lgb(X_train, y_train)

        proba_train = {
            "logreg": predict_proba_logistic(logreg, scaler, X_train),
            "xgb": predict_proba_tree(xgb_model, X_train),
            "lgb": predict_proba_tree(lgb_model, X_train),
        }
        meta_model = stack_predictions(proba_train, y_train=y_train, train_mode=True)

        # 用截至t-1嘅最後一行特徵, 預測t當日
        latest_X = X_train[-1:].copy()
        proba_input = {
            "logreg": predict_proba_logistic(logreg, scaler, latest_X),
            "xgb": predict_proba_tree(xgb_model, latest_X),
            "lgb": predict_proba_tree(lgb_model, latest_X),
        }
        ensemble_proba = stack_predictions(proba_input, meta_model=meta_model)[0]

        # ---- LSTM: 按頻率重訓, 其餘日子沿用舊模型 ----
        if days_since_lstm_retrain >= LSTM_RETRAIN_FREQ_DAYS:
            lstm_model = train_lstm(X_train, y_train, n_features=len(FEATURE_COLUMNS))
            days_since_lstm_retrain = 0
        else:
            days_since_lstm_retrain += 1

        lstm_proba = None
        if lstm_model is not None and len(X_train) >= SEQ_LEN:
            seq = X_train[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))
            lstm_proba = predict_lstm(lstm_model, seq)

        agree = agreement_score(ensemble_proba, lstm_proba)

        # ---- GARCH波幅預測(用截至t-1嘅歷史回報) ----
        garch_vol = fit_garch_forecast(train_slice["hsi_ret_1d"])

        # ---- 產生t日嘅OHLC預測 ----
        last_close = train_slice["hsi_close"].iloc[-1]
        atr = train_slice["atr14"].iloc[-1]
        overnight_gap_est = train_slice["overnight_gap"].rolling(10).mean().iloc[-1]
        ohlc_pred = predict_next_ohlc(last_close, overnight_gap_est, atr, ensemble_proba)

        direction = 1 if ensemble_proba > 0.5 else -1
        entry, stop, target = compute_entry_exit(
            ohlc_pred["expected_open"], atr, 1 if direction == 1 else 0
        )

        # ---- 關鍵: 攞返t日「真實」發生嘅結果嚟驗證預測 ----
        actual_open = current_row["hsi_open"]
        actual_high = current_row["hsi_high"]
        actual_low = current_row["hsi_low"]
        actual_close = current_row["hsi_close"]
        actual_direction = current_row["target_direction"]  # 注意: 呢個已經係shift(-1), 下面要調整
        actual_ret = current_row["hsi_ret_1d"]  # t日相對t-1嘅實際回報

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
            "actual_open": actual_open,
            "actual_high": actual_high,
            "actual_low": actual_low,
            "actual_close": actual_close,
            "actual_ret": actual_ret,
            "correct_direction": correct_direction,
        })

        print(f"模擬進度: {df.index[t_idx].date()} | proba_up={ensemble_proba:.3f} | "
              f"方向{'啱' if correct_direction else '錯'}")

    return pd.DataFrame(results).set_index("date")
