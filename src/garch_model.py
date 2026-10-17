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
        print(f"警告: GARCH擬合失敗 ({e}), 改用歷史標準

### Web Sources
1. Google Search: [yfinance download multi index columns best practice python](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCnTFmwey8wgXNnNsWUxki1-FlOEPgzAO4XuxI2JwgM7v3HgZ7eMd7X91BNQkqm91urGtf7RFQ35uIvd9LCrFc8DtTWcXdV7OABeDfH9bG_eP-QuOqM-tdMh4ZQwdEz9ot3tFnhQDj4vQIJXd6l_FgQlQZscoE5J2yS9yaZJLarDqR2zednvtAcqmSPHcl-PL4RvQwhRaWmHWWw-nGLaTQHiFlcaQxNjRsLoWA6XLuWKpXLInL-T3cOgucbw==)
2. [marketcalls.in](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHrkAn1RZ5OHG1FNOvb12KaiuJxKHeoILMe8GqRXat3yq4jpKEdyVVV3R7wHLWHjPfCRCJVb_lKBVKz0iWDavpS2CbYapwYYwBODZ2R6Vpfo1rlRCoAsH6hMIcD1OKpN6RLxtHpftOK_RtDFyux_erKqtz_9T98HkQGxkRtlUBRG4s3sNYq2rYNzhGIMReZKk7PPEdqckRPreUhDE0p0WvhizUdGQDSDEnvgfTucA5Z)
3. [stackoverflow.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHmcGiFY429QEPdMOJxLX1E-m5hFWE8q5eQQ5u4r7YmbsZLsIINeUWT7HprHDauHGSdInTcPXNJ-FqIkI5CqGuJ2cH4W2XN_wXw7NEXlNLZWBzJNPEPER3pJD_6c_OcVb-TSsvUB7aXqRbKiovhxbBVrUcrw4KUhl2ZhFgng1kciGRbVlxCPZtKsOUBJe27MUXGZlWReTU01vnjV_RuTNb3oliOxBG7IoU=)
4. [quantinsti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHxusbh2iV43Hvc9T-D39mhz6boIFzvrhWU45k0nlvaDwC7Lir8NfXOhemxy_G9QMhw91hPwkDYyfTu4RjrGv3VEPEe9SAaLA_UFfSYizGvzO84GtQvmWOsfE93v4d5EgUfzSiVQEy7is8rBtlVEpe3UE9bQGY7FLck00HMXok75wPVg5mu88RPlc_TgcT7PumLtxQHBTSUJ0GIaaZgWB40gfu-8w9MD9w6_--lF18Qrqgua_NCQWQYa0Qf9N883lzyzGE-9Q==)
