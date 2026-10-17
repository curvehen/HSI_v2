import numpy as np
import pandas as pd
from scipy.stats import norm
from config import DRIFT_WINDOW, DRIFT_MIN_SAMPLES, DRIFT_ACCURACY_FLOOR


def check_drift(performance_log):
    """
    檢查最近DRIFT_WINDOW筆預測嘅滾動準確度, 判斷係咪出現concept drift。
    回傳: dict with drift_detected(bool), rolling_accuracy, recommendation
    """
    if len(performance_log) < DRIFT_MIN_SAMPLES:
        return {
            "drift_detected": False,
            "rolling_accuracy": None,
            "n_samples": len(performance_log),
            "recommendation": f"樣本不足({len(performance_log)}筆), 未能判斷drift, 需最少{DRIFT_MIN_SAMPLES}筆"
        }

    recent = performance_log.tail(DRIFT_WINDOW)
    rolling_accuracy = recent["direction_correct"].mean()

    n = len(recent)
    se = np.sqrt(0.5 * 0.5 / n)
    z = (rolling_accuracy - 0.5) / se
    p_value = 1 - norm.cdf(z)

    drift_detected = rolling_accuracy < DRIFT_ACCURACY_FLOOR

    if drift_detected:
        recommendation = (
            f"警報: 最近{n}筆預測準確度降至{rolling_accuracy:.3f}, "
            f"低於警戒線{DRIFT_ACCURACY_FLOOR}, 建議觸發retrain或人手檢查特徵有效性"
        )
    elif rolling_accuracy < 0.52:
        recommendation = f"準確度{rolling_accuracy:.3f}偏弱但未達警戒線, 持續觀察"
    else:
        recommendation = f"準確度{rolling_accuracy:.3f}正常, 無需行動"

    return {
        "drift_detected": bool(drift_detected),
        "rolling_accuracy": round(float(rolling_accuracy), 4),
        "z_vs_random": round(float(z), 4),
        "p_value_vs_random": round(float(p_value), 6),
        "n_samples": n,
        "recommendation": recommendation
    }
