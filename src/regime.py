"""
市況regime簡易分類, 用VIX做門檻判斷(純量化規則, 唔經LLM, 唔經模型訓練)。
"""

def classify_regime(vix_level):
    """
    vix_level: float, 嚟自 ^VIX 最新收市值
    回傳: "LOW_VOL" / "NEUTRAL" / "HIGH_VOL" / "UNKNOWN"
    """
    if vix_level is None:
        return "UNKNOWN"
    if vix_level < 15:
        return "LOW_VOL"
    if vix_level <= 25:
        return "NEUTRAL"
    return "HIGH_VOL"
