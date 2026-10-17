from src.data_fetch import fetch_all
from src.features import build_features
from src.simulate_daily import run_daily_simulation
from src.stats_test import binomial_significance_test
from config import TARGET_LABEL
import os
import sys


def main():
    try:
        print(f"下載完整歷史數據 (目標: {TARGET_LABEL})...")
        raw = fetch_all(start="2015-01-01")
    except Exception as e:
        print(f"錯誤: 數據下載失敗 ({e})")
        sys.exit(1)

    df = build_features(raw)
    print(f"特徵建立完成, 共 {len(df)} 個有效交易日")

    try:
        print("開始逐日Point-in-Time模擬...")
        sim_results = run_daily_simulation(df)
    except Exception as e:
        print(f"錯誤: 模擬執行失敗 ({e})")
        sys.exit(1)

    os.makedirs("predictions", exist_ok=True)
    sim_results.to_csv(f"predictions/daily_simulation_{TARGET_LABEL}.csv")

    n_days = len(sim_results)
    n_correct = int(sim_results["correct_direction"].sum())
    win_rate = n_correct / n_days
    z, p = binomial_significance_test(n_days, n_correct, baseline_p=0.5)

    print(f"\n=== {TARGET_LABEL} 模擬結果 ===")
    print(f"總交易日數: {n_days}")
    print(f"勝率: {win_rate:.4f}")
    print(f"z值: {z}, p-value: {p}")
    print("結論:", "顯著優於隨機" if p and p < 0.05 else "未達統計顯著")


if __name__ == "__main__":
    main()
