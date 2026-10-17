import pandas as pd
import numpy as np


def classify_errors(performance_log):
    """
    將每日誤差拆做: 方向錯 / 幅度錯(高低區間) / 兩者皆錯
    """
    df = performance_log.copy()

    df["error_type"] = "correct"
    df.loc[(df["direction_correct"] == 0), "error_type"] = "direction_error"

    magnitude_threshold = 0.015  # high/low誤差超過1.5%視為幅度錯
    large_magnitude_error = (
        (df["high_error_pct"] > magnitude_threshold) |
        (df["low_error_pct"] > magnitude_threshold)
    )
    df.loc[(df["direction_correct"] == 1) & large_magnitude_error, "error_type"] = "magnitude_error"
    df.loc[(df["direction_correct"] == 0) & large_magnitude_error, "error_type"] = "direction_and_magnitude_error"

    return df


def summarize_error_attribution(performance_log):
    """
    輸出誤差類型分佈 + agreement分組嘅方向準確度比較
    """
    classified = classify_errors(performance_log)

    error_dist = classified["error_type"].value_counts(normalize=True).round(4)

    agreement_accuracy = None
    if "agreement" in classified.columns:
        agreement_accuracy = classified.groupby("agreement")["direction_correct"].agg(
            accuracy="mean", n="count"
        ).round(4)

    overall_accuracy = classified["direction_correct"].mean()
    avg_open_error = classified["open_error_pct"].mean()
    avg_close_error = classified["close_error_pct"].mean()

    return {
        "overall_direction_accuracy": round(overall_accuracy, 4),
        "avg_open_error_pct": round(avg_open_error, 5),
        "avg_close_error_pct": round(avg_close_error, 5),
        "error_type_distribution": error_dist.to_dict(),
        "agreement_breakdown": agreement_accuracy.to_dict("index") if agreement_accuracy is not None else None,
    }
