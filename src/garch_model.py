import numpy as np
from arch import arch_model

def fit_garch_forecast(returns_series, horizon=1):
    """
    用GARCH(1,1)對HSI日回報率建模, 預測下一日條件波動率
    returns_series: pandas Series of daily returns (in %)
    回傳: 預測嘅下一日標準差 (annualized -> daily)
    """
    r = returns_series.dropna() * 100  # arch套件慣用百分比單位
    model = arch_model(r, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=horizon)
    variance = forecast.variance.values[-1, -1]
    daily_vol = np.sqrt(variance) / 100  # 轉返小數
    return daily_vol
