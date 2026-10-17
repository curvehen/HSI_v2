import numpy as np
import pandas as pd
from config import WALK_FORWARD_TRAIN_WINDOW, WALK_FORWARD_TEST_WINDOW, WALK_FORWARD_STEP
from src.features import FEATURE_COLUMNS
from src.models import (train_logistic, train_xgb, train_lgb,
                          predict_proba_logistic, predict_proba_tree)
from src.lstm_model import train_lstm, make_sequences, predict_lstm, SEQ_LEN
from src.ensemble import stack_predictions, agreement_score
from src.garch_model import fit_garch_forecast

def walk_forward_run(df):
    """
    滾動窗口: train N日 -> test接下來M日 -> 向前移動Step日, 重複
    每一步都係真正out-of-sample, 避免look-ahead
    """
    results = []
    n = len(df)
    start = 0

    while start + WALK_FORWARD_TRAIN_WINDOW + WALK_FORWARD_TEST_WINDOW <= n:
        train_df = df.iloc[start : start + WALK_FORWARD_TRAIN_WINDOW]
        test_df = df.iloc[start + WALK_FORWARD_TRAIN_WINDOW :
                           start + WALK_FORWARD_TRAIN_WINDOW + WALK_FORWARD_TEST_WINDOW]

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["target_direction"].values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df["target_direction"].values

        # 訓練三個base模型
        logreg, scaler = train_logistic(X_train, y_train)
        xgb_model = train_xgb(X_train, y_train)
        lgb_model = train_lgb(X_train, y_train)

        proba_train = {
            "logreg": predict_proba_logistic(logreg, scaler, X_train),
            "xgb": predict_proba_tree(xgb_model, X_train),
            "lgb": predict_proba_tree(lgb_model, X_train),
        }
        meta_model = stack_predictions(proba_train, y_train=y_train, train_mode=True)

        proba_test = {
            "logreg": predict_proba_logistic(logreg, scaler, X_test),
            "xgb": predict_proba_tree(xgb_model, X_test),
            "lgb": predict_proba_tree(lgb_model, X_test),
        }
        ensemble_proba_test = stack_predictions(proba_test, meta_model=meta_model)

        # LSTM獨立對齊層 (用同一訓練窗口)
        lstm_model = train_lstm(X_train, y_train, n_features=len(FEATURE_COLUMNS))
        lstm_probas = []
        if lstm_model is not None:
            full_X = np.vstack([X_train, X_test])
            for i in range(len(X_test)):
                seq_end = len(X_train) + i
                seq_start = seq_end - SEQ_LEN
                if seq_start < 0:
                    lstm_probas.append(None)
                    continue
                seq = full_X[seq_start:seq_end].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))
                lstm_probas.append(predict_lstm(lstm_model, seq))
        else:
            lstm_probas = [None] * len(X_test)

        # GARCH波幅預測 (用train窗口嘅歷史回報)
        garch_vol = fit_garch_forecast(train_df["hsi_ret_1d"])

        for i, idx in enumerate(test_df.index):
            agree = agreement_score(ensemble_proba_test[i], lstm_probas[i])
            results.append({
                "date": idx,
                "ensemble_proba": ensemble_proba_test[i],
                "lstm_proba": lstm_probas[i],
                "agreement": agree,
                "garch_vol_forecast": garch_vol,
                "target_direction": y_test[i],
                "target_ret": test_df["target_ret"].iloc[i],
                "hsi_close": test_df["hsi_close"].iloc[i],
                "atr14": test_df["atr14"].iloc[i],
                "overnight_gap": test_df["overnight_gap"].iloc[i],
            })

        start += WALK_FORWARD_STEP

    return pd.DataFrame(results).set_index("date")
