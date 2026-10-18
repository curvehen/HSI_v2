"""
每日執行入口: 對watchlist全部ticker跑預測, 組裝context_data,
呼叫LLM生成分析, 回填/記錄hit-rate, 最後發送email。
"""

import json
import os
from datetime import datetime

from config import WATCHLIST, MACRO_TICKERS, SECTOR_MAP
from src.data_fetch import fetch_ohlcv, fetch_latest_close
from src.features import build_features
from src.ensemble import load_or_train_ensemble, ensemble_predict_proba
from src.lstm_model import load_or_train_lstm, lstm_predict_proba
from src.garch_model import forecast_volatility
from src.risk import predict_next_ohlc, compute_entry_exit, risk_reward_ratio
from src.candlestick_patterns import detect_patterns
from src.regime import classify_regime
from src.llm_prompts import build_stock_prompt
from src.llm_analysis import generate_analysis
from src.hit_rate import resolve_pending_predictions, log_new_prediction, get_hit_rate
from src.email_sender import send_prediction_email

PREDICTIONS_DIR = "predictions"
OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, "watchlist_prediction.json")


def agreement_score(ensemble_direction, lstm_direction):
    return "agree" if ensemble_direction == lstm_direction else "disagree"


def confidence_filter_pass(proba_up, agreement):
    """
    信心過濾規則: 機率需夠極端(偏離0.5足夠遠), 且Ensemble與LSTM方向一致, 才算通過。
    閾值可按你哋之前Champion-Challenger驗證結果調整。
    """
    distance = abs(proba_up - 0.5)
    return agreement == "agree" and distance >= 0.15


def build_macro_context():
    """抓取macro tickers嘅最新數值, 組成context_data嘅市場背景部分"""
    macro = {}
    for ticker in MACRO_TICKERS:
        try:
            macro[ticker] = fetch_latest_close(ticker)
        except Exception as e:
            print(f"警告: 無法取得 {ticker} 最新數據 ({e})")
            macro[ticker] = None
    return macro


def predict_single_target(target_name, ticker, macro_context, run_date):
    try:
        df = fetch_ohlcv(ticker)
        features_df = build_features(df, macro_context=macro_context)

        ensemble_model = load_or_train_ensemble(ticker)
        proba_up_ensemble = ensemble_predict_proba(ensemble_model, features_df)
        ensemble_direction = "up" if proba_up_ensemble >= 0.5 else "down"

        lstm_model = load_or_train_lstm(ticker)
        proba_up_lstm = lstm_predict_proba(lstm_model, features_df)
        lstm_direction = "up" if proba_up_lstm >= 0.5 else "down"

        agreement = agreement_score(ensemble_direction, lstm_direction)
        passed_filter = confidence_filter_pass(proba_up_ensemble, agreement)

        atr = forecast_volatility(df)
        last_close = float(df["Close"].iloc[-1])

        predicted_ohlc = predict_next_ohlc(
            last_close=last_close, proba_up=proba_up_ensemble, atr=atr
        )
        entry, stop, target_price = compute_entry_exit(
            last_close=last_close, predicted_ohlc=predicted_ohlc, proba_up=proba_up_ensemble
        )
        rr_ratio = risk_reward_ratio(entry, stop, target_price)

        rsi14 = float(features_df["RSI14"].iloc[-1]) if "RSI14" in features_df.columns else None
        patterns = detect_patterns(df)

        regime = classify_regime(macro_context.get("^VIX")) if target_name == "HSI" else None

        context_data = {
            "target": target_name,
            "ticker": ticker,
            "run_date": run_date,
            "last_known_close": last_close,
            "ensemble_proba_up": proba_up_ensemble,
            "lstm_proba_up": proba_up_lstm,
            "agreement": agreement,
            "confidence_filter_pass": passed_filter,
            "predicted_ohlc": predicted_ohlc,
            "suggested_entry": entry,
            "suggested_stop_loss": stop,
            "suggested_take_profit": target_price,
            "risk_reward_ratio": rr_ratio,
            "ATR14": atr,
            "RSI14": rsi14,
            "candlestick_patterns": patterns,
            "regime": regime,
            "macro_context": macro_context,
        }

        prompt = build_stock_prompt(context_data)
        llm_text = generate_analysis(prompt)
        context_data["llm_analysis"] = llm_text

        hit_rate_str, hit_rate_n = get_hit_rate(ticker)
        context_data["hit_rate"] = hit_rate_str
        context_data["hit_rate_samples"] = hit_rate_n

        log_new_prediction(target_name, ticker, proba_up_ensemble, last_close, run_date=run_date)

        return context_data

    except Exception as e:
        print(f"錯誤: {target_name} ({ticker}) 預測失敗 - {e}")
        return {"target": target_name, "ticker": ticker, "error": str(e)}


def run():
    run_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    print("步驟1: 抓取macro context...")
    macro_context = build_macro_context()

    print("步驟2: 回填上一輪未結算嘅預測(hit-rate)...")
    current_prices = {}
    for target_name, ticker in WATCHLIST.items():
        try:
            current_prices[ticker] = fetch_latest_close(ticker)
        except Exception as e:
            print(f"警告: 無法取得 {ticker} 最新價格做回填 ({e})")
    resolve_pending_predictions(current_prices)

    print("步驟3: 逐個target執行預測...")
    results = []
    for target_name, ticker in WATCHLIST.items():
        print(f"  -> 處理 {target_name} ({ticker})")
        result = predict_single_target(target_name, ticker, macro_context, run_date)
        results.append(result)

    output = {"run_date": run_date, "results": results}

    print(f"步驟4: 儲存結果至 {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("步驟5: 發送email...")
    send_prediction_email(output)

    print("完成。")
    return output


if __name__ == "__main__":
    run()
