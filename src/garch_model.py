import numpy as np
from arch import arch_model


def fit_garch_forecast(returns_series, horizon=1, fallback_vol=0.01):
    """
    用GARCH(1,1)預測下一日條件波動率。
    若數據不足或擬合失敗, 回傳fallback值(用歷史標準差估算), 唔會crash。
    """
    r = returns_series.dropna()
    if len(r) < 100:
        return float(r.std()) if len(r) > 1 else fallback_vol

    try:
        r_pct = r * 100
        model = arch_model(r_pct, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=horizon)
        variance = forecast.variance.values[-1, -1]
        daily_vol = np.sqrt(variance) / 100
        if np.isnan(daily_vol) or daily_vol <= 0:
            return float(r.std())
        return float(daily_vol)
    except Exception as e:
        print(f"警告: GARCH擬合失敗 ({e}), 改用歷史標準差代替")
        return float(r.std())
