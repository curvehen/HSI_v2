"""
批量預測: 對config.WATCHLIST每隻ticker重複執行predict.py已驗證嘅單一目標pipeline,
組裝HSI獨立報告 + 個股表格, 加LLM分析, 維護命中率log, 觸發email發送。

已確認無誤的依賴:
- src.risk.compute_entry_exit / predict_next_ohlc 簽名已對照risk.py原始碼確認
- src.llm_analysis 三個function(generate_hsi_analysis / generate_stock_analysis /
  generate_watchlist_summary)內部自行組prompt, 呼叫方只需傳payload/results, 不需自己build prompt
"""

import json
import os
from datetime import datetime

from src.data_fetch import fetch_all
from src.features import build_features, FEATURE_COLUMNS
from src.models import (train_logistic, train_xgb, train_lgb,
                         predict_proba_logistic, predict_proba_tree)
from src.lstm_model import train_lstm, predict_lstm, SEQ_LEN
from src.ensemble import stack_predictions, agreement_score
from src.garch_model import fit_garch_forecast
from src.risk import predict_next_ohlc, compute_entry_exit
from src.hit_rate import resolve_pending_predictions, log_new_prediction, get_hit_rate
from config import WATCHLIST, HSI_TICKER, SECTOR_MAP

try:
    from src.llm_analysis import (generate_hsi_analysis,
                                   generate_stock_analysis,
                                   generate_watchlist_summary)
    LLM_AVAILABLE = True
except Exception as e:
    print(f"警告: 無法載入 src.llm_analysis ({e}), LLM分析將跳過")
    LLM_AVAILABLE = False

try:
    from src.email_sender import send_prediction_email
    EMAIL_AVAILABLE = True
except Exception as e:
    print(f"警告: 無法載入 src.email_sender ({e}), email發送將跳過")
    EMAIL_AVAILABLE = False

PRED_DIR = "predictions"
OUTPUT_PATH = os.path.join(PRED_DIR, "watchlist_prediction.json")


def classify_regime(vix_level):
    if vix_level is None or vix_level != vix_level:  # NaN check
        return "UNKNOWN"
    if vix_level < 15:
        return "LOW_VOL"
    if vix_level <= 25:
        return "NEUTRAL"
    return "HIGH_VOL"


def classify_scenario_context():
    wd = datetime.now().weekday()  # Mon=0 ... Sun=6
    return "週末" if wd in (4, 5, 6) else "平日"


def rate_risk_reward(rr_ratio):
    """風險回報比評級文字, 門檻可自行調整(非已confirm事實, 純量化規則)"""
    if rr_ratio is None:
        return "—"
    if rr_ratio >= 2.0:
        return "優"
    if rr_ratio >= 1.5:
        return "良"
    if rr_ratio >= 1.0:
        return "一般"
    return "差"


def predict_single_target(target_label, ticker):
    """
    對單一ticker跑完整pipeline, 回傳result dict。
    任何例外喺呢層catch, 回傳帶error欄位嘅dict, 唔會令成個watchlist中斷。
    """
    try:
        benchmark = HSI_TICKER
        raw = fetch_all(target_ticker=ticker, benchmark_ticker=benchmark)
        df = build_features(raw)

        if len(df) < 100:
            return {"target": target_label, "ticker": ticker,
                    "error": f"可用數據不足({len(df)}日)"}

        resolve_pending_predictions(df, ticker)

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
        if overnight_gap_est != overnight_gap_est:
            overnight_gap_est = 0.0

        ohlc_pred = predict_next_ohlc(last_close, overnight_gap_est, atr, ensemble_proba)

        direction = 1 if ensemble_proba > 0.5 else -1
        entry, stop, target = compute_entry_exit(
            ohlc_pred["expected_open"], atr, 1 if direction == 1 else 0)

        strong_confidence = ensemble_proba > 0.55 or ensemble_proba < 0.45
        confidence_filter_pass = bool(strong_confidence and agree != "disagree")

        rr_ratio = None
        try:
            risk_amt = abs(entry - stop)
            reward_amt = abs(target - entry)
            rr_ratio = round(reward_amt / risk_amt, 2) if risk_amt > 0 else None
        except Exception:
            pass

        vix_level = float(raw["vix_close"].iloc[-1]) if "vix_close" in raw.columns else None
        regime = classify_regime(vix_level)

        rsi_latest = float(df["rsi14"].iloc[-1])
        macd_hist_latest = float(df["macd_hist"].iloc[-1])
        ret_1d_latest = float(df["target_ret_1d"].iloc[-1]) if "target_ret_1d" in df.columns else None

        run_date = datetime.now().strftime("%Y-%m-%d")
        log_new_prediction(target_label, ticker, run_date, direction, last_close)
        hit_rate_str, hit_rate_n = get_hit_rate(ticker)

        result = {
            "target": target_label,
            "ticker": ticker,
            "sector": SECTOR_MAP.get(ticker, "—"),
            "run_date": run_date,
            "last_known_close": round(last_close, 3),
            "ensemble_proba_up": round(ensemble_proba, 4),
            "lstm_proba_up": round(lstm_proba, 4) if lstm_proba is not None else None,
            "agreement": agree,
            "regime": regime,
            "garch_daily_vol_forecast": round(float(garch_vol), 5),
            "predicted_ohlc": {k: round(v, 3) for k, v in ohlc_pred.items()},
            "ret_1d": round(ret_1d_latest, 4) if ret_1d_latest is not None else None,
            "suggested_entry": round(float(entry), 3),
            "suggested_stop_loss": round(float(stop), 3),
            "suggested_take_profit": round(float(target), 3),
            "risk_reward_ratio": rr_ratio,
            "risk_reward_rating": rate_risk_reward(rr_ratio),
            "confidence_filter_pass": confidence_filter_pass,
            "hit_rate": hit_rate_str,
            "hit_rate_samples": hit_rate_n,
            "RSI14": round(rsi_latest, 1),
            "context_data": {
                "RSI14": round(rsi_latest, 1),
                "MACD_hist": round(macd_hist_latest, 4),
                "news_headlines": [],  # news_fetch整合待確認實際function簽名, 暫留空list
            },
        }
        return result

    except Exception as e:
        return {"target": target_label, "ticker": ticker, "error": str(e)}


def attach_llm_analysis(results):
    """為每個result加llm_analysis文字欄位。LLM唔可用時跳過, 唔會令pipeline中斷。"""
    if not LLM_AVAILABLE:
        for r in results:
            r["llm_analysis"] = None
        return results

    for r in results:
        if r.get("error"):
            r["llm_analysis"] = None
            continue
        try:
            if r["target"] == "HSI":
                r["llm_analysis"] = generate_hsi_analysis(r)
            else:
                r["llm_analysis"] = generate_stock_analysis(r)
        except Exception as e:
            print(f"警告: {r['target']} LLM分析失敗 ({e})")
            r["llm_analysis"] = None
    return results


def main():
    print(f"開始批量預測, 共 {len(WATCHLIST)} 個目標...")
    results = []

    for target_label, ticker in WATCHLIST.items():
        print(f"處理中: {target_label} ({ticker})...")
        result = predict_single_target(target_label, ticker)
        if result.get("error"):
            print(f"  警告: {target_label} 處理失敗 ({result['error']})")
        else:
            print(f"  完成: P(升)={result['ensemble_proba_up']:.4f}")
        results.append(result)

    print("附加個股/HSI LLM分析...")
    results = attach_llm_analysis(results)

    watchlist_summary = None
    if LLM_AVAILABLE:
        try:
            print("生成watchlist彙總LLM分析...")
            watchlist_summary = generate_watchlist_summary(results)
        except Exception as e:
            print(f"警告: watchlist彙總LLM分析失敗 ({e})")

    output = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "scenario_context": classify_scenario_context(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "watchlist_summary": watchlist_summary,
        "results": results,
    }

    os.makedirs(PRED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n預測結果已存至 {OUTPUT_PATH}")

    if EMAIL_AVAILABLE:
        try:
            send_prediction_email(output)
            print("Email已發送")
        except Exception as e:
            print(f"警告: Email發送失敗 ({e})")
    else:
        print("警告: email模組不可用, 已跳過發送")


if __name__ == "__main__":
    main()
