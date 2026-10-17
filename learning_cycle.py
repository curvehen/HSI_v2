from src.data_fetch import fetch_all
from src.features import build_features
from src.performance_tracker import evaluate_pending_predictions
from src.error_attribution import summarize_error_attribution
from src.drift_detection import check_drift
from src.hyperparam_search import run_hyperparameter_search
from src.champion_challenger import run_champion_challenger_evaluation
from config import PERFORMANCE_LOG_PATH, HYPEROPT_MIN_HISTORY
import pandas as pd
import json
import os
import sys


def main():
    print("=== 自我學習週期開始 ===\n")

    print("Step 1: 下載最新數據...")
    raw = fetch_all()
    df = build_features(raw)

    print("\nStep 2: 評估過往預測嘅實際結果...")
    actual_df = raw[["hsi_open", "hsi_high", "hsi_low", "hsi_close"]]
    performance_log = evaluate_pending_predictions(actual_df)

    if performance_log is None or len(performance_log) == 0:
        print("冇足夠歷史預測紀錄, 跳過誤差歸因同drift檢查, 等累積更多predict.py執行紀錄")
    else:
        print("\nStep 3: 誤差歸因分析...")
        attribution = summarize_error_attribution(performance_log)
        print(json.dumps(attribution, indent=2, ensure_ascii=False))

        print("\nStep 4: Drift偵測...")
        drift_result = check_drift(performance_log)
        print(json.dumps(drift_result, indent=2, ensure_ascii=False))

        if drift_result["drift_detected"]:
            print("\n警報已觸發: 模型表現顯著衰退, 建議進行以下步驟5-6嘅retrain流程")

    if len(df) < HYPEROPT_MIN_HISTORY:
        print(f"\n數據量不足({len(df)}日), 跳過hyperparameter search (需要最少{HYPEROPT_MIN_HISTORY}日)")
        sys.exit(0)

    print("\nStep 5: Hyperparameter搜索 (Optuna, 受guardrail約束)...")
    challenger_params = run_hyperparameter_search(df)

    print("\nStep 6: Champion-Challenger驗證...")
    cc_result = run_champion_challenger_evaluation(df)
    print(json.dumps(cc_result, indent=2, ensure_ascii=False))

    if cc_result.get("should_promote"):
        print("\n>>> 建議: Challenger表現顯著優於Champion <<<")
        print(">>> 此步驟需人手覆核確認, 未自動執行升級 <<<")
        print(">>> 如確認升級, 請手動執行: python -c \"from src.champion_challenger import promote_challenger_to_champion; promote_challenger_to_champion()\" <<<")
    else:
        print("\n維持現有champion參數, 未達升級門檻")

    os.makedirs("predictions", exist_ok=True)
    summary = {
        "run_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "performance_summary": attribution if performance_log is not None and len(performance_log) > 0 else None,
        "drift_result": drift_result if performance_log is not None and len(performance_log) > 0 else None,
        "champion_challenger_result": cc_result,
    }
    with open("predictions/learning_cycle_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n=== 自我學習週期完成, 結果已存至 predictions/learning_cycle_summary.json ===")


if __name__ == "__main__":
    main()
