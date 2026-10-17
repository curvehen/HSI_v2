from src.data_fetch import fetch_all
from src.features import build_features, FEATURE_COLUMNS
from src.models import train_logistic, train_xgb, train_lgb, predict_proba_logistic, predict_proba_tree
from src.lstm_model import train_lstm, predict_lstm, SEQ_LEN
from src.ensemble import stack_predictions, agreement_score
from src.garch_model import fit_garch_forecast
from src.risk import predict_next_ohlc, compute_entry_exit
import json
import os
import sys
from datetime import datetime


def main():
    try:
        raw = fetch_all()
    except Exception as e:
        print(f"錯誤: 數據下載失敗 ({e})")
        sys.exit(1)

    df = build_features(raw)
    if len(df) < 100:
        print(f"錯誤: 可用數據不足({len(df)}日), 無法訓練模型")
        sys.exit(1)

    X = df[FEATURE_COLUMNS].values
    y = df["target_direction"].values

    try:
        logreg, scaler = train_logistic(X, y)
        xgb_model = train_xgb(X, y)
        lgb_model = train_lgb(X, y)
    except Exception as e:
        print(f"錯誤: 模型訓練失敗 ({e})")
        sys.exit(1)

    proba_train = {
        "logreg": predict_proba_logistic(logreg, scaler, X),
        "xgb": predict_proba_tree(xgb_model, X),
        "lgb": predict_proba_tree(lgb_model, X),
    }
    meta_model = stack_predictions(proba_train, y_train=y, train_mode=True)

    latest_X = X[-1:].copy()
    proba_latest = {
        "logreg": predict_proba_logistic(logreg, scaler, latest_X),
        "xgb": predict_proba_tree(xgb_model, latest_X),
        "lgb": predict_proba_tree(lgb_model, latest_X),
    }
    ensemble_proba = float(stack_predictions(proba_latest, meta_model=meta_model)[0])

    lstm_model = train_lstm(X, y, n_features=len(FEATURE_COLUMNS))
    lstm_proba = None
    if lstm_model is not None and len(X) >= SEQ_LEN:
        seq = X[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))
        lstm_proba = predict_lstm(lstm_model, seq)

    agree = agreement_score(ensemble_proba, lstm_proba)
    garch_vol = fit_garch_forecast(df["hsi_ret_1d"])

    last_close = float(df["hsi_close"].iloc[-1])
    atr = float(df["atr14"].iloc[-1])
    overnight_gap_est = df["overnight_gap"].rolling(10).mean().iloc[-1]
    if overnight_gap_est != overnight_gap_est:  # NaN check
        overnight_gap_est = 0.0

    ohlc_pred = predict_next_ohlc(last_close, overnight_gap_est, atr, ensemble_proba)

    direction = 1 if ensemble_proba > 0.5 else -1
    entry, stop, target = compute_entry_exit(ohlc_pred["expected_open"], atr, 1 if direction == 1 else 0)

    output = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "last_known_close": round(last_close, 1),
        "ensemble_proba_up": round(ensemble_proba, 4),
        "lstm_proba_up": round(lstm_proba, 4) if lstm_proba is not None else None,
        "agreement": agree,
        "garch_daily_vol_forecast": round(float(garch_vol), 5),
        "predicted_ohlc": ohlc_pred,
        "suggested_entry": round(float(entry), 1),
        "suggested_stop_loss": round(float(stop), 1),
        "suggested_take_profit": round(float(target), 1),
        "strong_confidence = ensemble_proba > 0.55 or ensemble_proba < 0.45
        confidence_filter_pass = bool(strong_confidence and agree != "disagree")
    }

        print(json.dumps(output, indent=2, ensure_ascii=False))

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/latest_prediction.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    from src.performance_tracker import log_prediction
    log_prediction(output)

    print("\n預測已存至 predictions/latest_prediction.json 同 prediction_history.csv")




if __name__ == "__main__":
    main()
