from src.predict_single import predict_single_ticker
from src.performance_tracker import log_prediction
from config import WATCHLIST, BENCHMARK_TICKER, START_DATE
import json
import os
import sys
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
        else:
            print(f"proba_up={result['ensemble_proba_up']} | "
                  f"agreement={result['agreement']} | "
                  f"filter_pass={result['confidence_filter_pass']}\n")

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

    output = {"run_date": run_date, "results": all_results}

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/watchlist_prediction.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== 批量預測完成 ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n結果已存至 predictions/watchlist_prediction.json")

    failed = [r for r in all_results if r.get("error")]
    if failed:
        print(f"\n警告: {len(failed)} 個標的處理失敗: {[r['target'] for r in failed]}")


if __name__ == "__main__":
    main()
