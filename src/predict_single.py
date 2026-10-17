import numpy as np
from src.data_fetch import fetch_all
from src.features import build_features, FEATURE_COLUMNS
from src.models import train_logistic, train_xgb, train_lgb, predict_proba_logistic, predict_proba_tree
from src.lstm_model import train_lstm, predict_lstm, SEQ_LEN
from src.ensemble import stack_predictions, agreement_score
from src.garch_model import fit_garch_forecast
from src.risk import predict_next_ohlc, compute_entry_exit


def predict_single_ticker(label, ticker, benchmark_ticker="^HSI", start="2015-01-01"):
    """
    對單一股票/指數跑完整預測pipeline, 回傳結果dict。
    失敗時回傳帶error key嘅dict, 唔會中斷其他股票嘅批量處理。
    """
    try:
        raw = fetch_all(start=start, target_ticker=ticker, benchmark_ticker=benchmark_ticker)
        df = build_features(raw)

        if len(df) < 100:
            return {"target": label, "ticker": ticker, "error": f"數據不足({len(df)}日)"}

        X = df[FEATURE_COLUMNS].values
        y = df["target_direction"].values

        logreg, scaler = train_logistic(X, y)
        xgb_model = train_xgb(X, y)
        lgb_model = train_lgb(X, y)

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
        garch_vol = fit_garch_forecast(df["target_ret_1d"])

        last_close = float(df["target_close"].iloc[-1])
        atr = float(df["atr14"].iloc[-1])
        overnight_gap_est = df["overnight_gap"].rolling(10).mean().iloc[-1]
        if overnight_gap_est != overnight_gap_est:  # NaN check
            overnight_gap_est = 0.0

        ohlc_pred = predict_next_ohlc(last_close, overnight_gap_est, atr, ensemble_proba)
        direction = 1 if ensemble_proba > 0.5 else -1
        entry, stop, target = compute_entry_exit(
            ohlc_pred["expected_open"], atr, 1 if direction == 1 else 0
        )

        strong_confidence = ensemble_proba > 0.55 or ensemble_proba < 0.45
        confidence_filter_pass = bool(strong_confidence and agree != "disagree")

        # ---- 補充context_data, 供LLM分析使用 ----
        latest_row = df.iloc[-1]

        def _safe_float(val, default=0.0):
            try:
                f = float(val)
                return f if f == f else default  # NaN check
            except (TypeError, ValueError):
                return default

        context_data = {
            "美股隔夜_nasdaq_ret": round(_safe_float(latest_row.get("nasdaq_ret")), 5),
            "美股隔夜_sp500_ret": round(_safe_float(latest_row.get("sp500_ret")), 5),
            "美股隔夜_dow_ret": round(_safe_float(latest_row.get("dow_ret")), 5),
            "人民幣匯率變動_usdcnh_ret": round(_safe_float(latest_row.get("usdcnh_ret_proxy")), 5),
            "權重股動態_topweight_avg_ret": round(_safe_float(latest_row.get("topweight_avg_ret")), 5),
            "權重股成交量異動_topweight_volume_ratio": round(_safe_float(latest_row.get("topweight_volume_ratio"), 1.0), 3),
            "南向資金_官方數據": None,
            "南向資金_AH溢價代理指標": round(_safe_float(latest_row.get("ah_premium_change_proxy")), 5),
            "恒生科技相對強弱_proxy": round(_safe_float(latest_row.get("tech_relative_strength_proxy")), 5),
            "VIX恐慌指數": round(_safe_float(latest_row.get("vix_level")), 2),
            "美元指數變動_dxy_ret": round(_safe_float(latest_row.get("dxy_ret")), 5),
            "美債10年孳息變動_us10y_ret": round(_safe_float(latest_row.get("us10y_ret")), 5),
            "恒指期貨隱含開盤": None,
            "重要消息催化": None,
            "RSI14": round(_safe_float(latest_row.get("rsi14")), 2),
            "MACD": round(_safe_float(latest_row.get("macd")), 4),
            "MACD_signal": round(_safe_float(latest_row.get("macd_signal")), 4),
            "MACD_hist": round(_safe_float(latest_row.get("macd_hist")), 4),
            "布林帶位置_bb_position": round(_safe_float(latest_row.get("bb_position")), 4),
            "布林帶寬度_bb_width": round(_safe_float(latest_row.get("bb_width")), 4),
            "成交量異動_vol_ratio": round(_safe_float(latest_row.get("vol_ratio"), 1.0), 3),
            "ATR14": round(atr, 2),
            "ATR百分比_atr_pct": round(_safe_float(latest_row.get("atr_pct")), 5),
        }

        return {
            "target": label,
            "ticker": ticker,
            "last_known_close": round(last_close, 2),
            "ensemble_proba_up": round(ensemble_proba, 4),
            "lstm_proba_up": round(lstm_proba, 4) if lstm_proba is not None else None,
            "agreement": agree,
            "garch_daily_vol_forecast": round(float(garch_vol), 5),
            "predicted_ohlc": ohlc_pred,
            "suggested_entry": round(float(entry), 2),
            "suggested_stop_loss": round(float(stop), 2),
            "suggested_take_profit": round(float(target), 2),
            "confidence_filter_pass": confidence_filter_pass,
            "context_data": context_data,
            "error": None,
        }

    except Exception as e:
        return {"target": label, "ticker": ticker, "error": str(e)}
