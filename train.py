from src.data_fetch import fetch_all
from src.features import build_features
from src.walk_forward import walk_forward_run
from src.backtest import run_backtest, compute_metrics
from src.stats_test import binomial_significance_test
import pandas as pd
import os
from config import PRED_DIR

def main():
    print("Step 1/4: 下載數據...")
    raw = fetch_all()

    print("Step 2/4: 建立特徵...")
    df = build_features(raw)

    print("Step 3/4: Walk-forward訓練+驗證 (會花幾分鐘, 包含LSTM訓練)...")
    wf_results = walk_forward_run(df)

    print("Step 4/4: Backtest + 顯著性檢定...")
    full_df, traded = run_backtest(wf_results, threshold=0.55)
    metrics = compute_metrics(traded)

    print("\n=== Walk-Forward Out-of-Sample 結果 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if metrics["n_trades"]:
        z, p = binomial_significance_test(metrics["n_trades"], metrics["wins"], baseline_p=0.5)
        print(f"\nz值: {z}, p-value: {p}")
        print("結論:", "顯著高於隨機 (p<0.05)" if p and p < 0.05 else "未達統計顯著, 不可盡信")

    agree_breakdown = full_df.groupby("agreement")["trade_ret"].agg(["mean", "count"])
    print("\n=== LSTM對齊層分組表現 (驗證對齊機制是否有用) ===")
    print(agree_breakdown)

    os.makedirs(PRED_DIR, exist_ok=True)
    full_df.to_csv(f"{PRED_DIR}/walk_forward_results.csv")
    print(f"\n完整結果已存至 {PRED_DIR}/walk_forward_results.csv")

if __name__ == "__main__":
    main()
