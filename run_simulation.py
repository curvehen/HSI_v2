from src.data_fetch import fetch_all
from src.features import build_features
from src.simulate_daily import run_daily_simulation
from src.stats_test import binomial_significance_test
import numpy as np

def main():
    print("下載完整歷史數據...")
    raw = fetch_all(start="2015-01-01")
    df = build_features(raw)

    print("開始逐日Point-in-Time模擬 (2026-01-01 起)...")
    sim_results = run_daily_simulation(df)

    sim_results.to_csv("predictions/daily_simulation_2026.csv")

    n_days = len(sim_results)
    n_correct = sim_results["correct_direction"].sum()
    win_rate = n_correct / n_days

    z, p = binomial_significance_test(n_days, n_correct, baseline_p=0.5)

    print(f"\n=== 2026-01-01 至今 模擬結果 ===")
    print(f"總交易日數: {n_days}")
    print(f"方向預測正確數: {n_correct}")
    print(f"勝率: {win_rate:.4f}")
    print(f"z值: {z}, p-value: {p}")
    print("結論:", "顯著優於隨機" if p and p < 0.05 else "未達統計顯著")

if __name__ == "__main__":
    main()
