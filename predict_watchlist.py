from src.predict_single import predict_single_ticker
from src.performance_tracker import log_prediction
from src.llm_analysis import generate_hsi_analysis, generate_stock_analysis
from config import WATCHLIST, BENCHMARK_TICKER, START_DATE, ENABLE_LLM_ANALYSIS
import json
import os
from datetime import datetime


def main():
    run_date = datetime.now().strftime("%Y-%m-%d")
    all_results = []

    print(f"=== 批量預測開始: {len(WATCHLIST)} 個標的 ===\n")

    for label, ticker in WATCHLIST.items():
        print(f"--- 處理: {label} ({ticker}) ---")
        result = predict_single_ticker(
            label=label, ticker=ticker,
            benchmark_ticker=BENCHMARK_TICKER, start=START_DATE
        )

        if result.get("error"):
            print(f"失敗: {result['error']}\n")
            all_results.append(result)
            continue

        print(f"proba_up={result['ensemble_proba_up']} | "
              f"agreement={result['agreement']} | "
              f"filter_pass={result['confidence_filter_pass']}")

        if ENABLE_LLM_ANALYSIS:
            print("正在生成LLM分析...")
            try:
                if label == "HSI":
                    result["llm_analysis"] = generate_hsi_analysis(result)
                else:
                    result["llm_analysis"] = generate_stock_analysis(result)
            except Exception as e:
                print(f"警告: LLM分析失敗 ({e})")
                result["llm_analysis"] = f"[LLM分析生成失敗: {e}]"
        else:
            result["llm_analysis"] = None

        log_record = {
            "run_date": run_date,
            "target": label,
            "ensemble_proba_up": result["ensemble_proba_up"],
            "agreement": result["agreement"],
            "predicted_open": result["predicted_ohlc"]["expected_open"],
            "predicted_high": result["predicted_ohlc"]["expected_high"],
            "predicted_low": result["predicted_ohlc"]["expected_low"],
            "predicted_close": result["predicted_ohlc"]["expected_close"],
        }
        log_prediction(log_record)

        all_results.append(result)
        print()

    output = {"run_date": run_date, "results": all_results}

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/watchlist_prediction.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== 批量預測完成 ===")
    print(f"結果已存至 predictions/watchlist_prediction.json")

    failed = [r["target"] for r in all_results if r.get("error")]
    if failed:
        print(f"\n警告: {len(failed)} 個標的處理失敗: {failed}")


if __name__ == "__main__":
    main()
