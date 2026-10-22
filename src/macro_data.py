"""
宏觀輔助數據抓取模組：南向資金、恒生科技期指隱含開盤估算。
這些數據並非HKEX官方即時API輸出，使用時須在LLM prompt/報告中註明來源與局限性。
"""

import pandas as pd

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

import numpy as np
import yfinance as yf


def get_southbound_flow():
    """
    獲取最新南向資金（港股通滬+深合計）淨流入數據。
    數據源: akshare -> 東方財富網公開頁面 (免費, 非HKEX官方API)。
    失敗或套件未安裝時回傳 None，交由prompt誠實標註「數據未提供」。
    """
    if not AKSHARE_AVAILABLE:
        return None
    try:
        df = ak.stock_hsgt_hist_em(symbol="南向資金")
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        net_col = [c for c in df.columns if "淨買額" in c or "淨流入" in c]
        value = float(latest[net_col[0]]) if net_col else None
        return {
            "date": str(latest.get("日期", latest.name)),
            "net_inflow_rmb": round(value, 1) if value is not None else None,
            "source": "akshare/東方財富網公開數據(非HKEX官方即時API)",
        }
    except Exception as e:
        print(f"警告: 無法取得南向資金數據 ({e})")
        return None


def estimate_hstech_implied_open(lookback_days=90):
    """
    用美股夜盤相關ETF(KWEB, 追蹤中概股/港股科技股)對HSTECH的歷史Beta關係,
    估算恒生科技指數期貨隱含開盤方向與幅度。
    此為「模擬估算值」, 非HKEX官方計算開盤價(COP), 使用時必須標註此聲明。
    """
    try:
        hstech = yf.download("^HSTECH", period=f"{lookback_days}d",
                              interval="1d", progress=False)["Close"]
        kweb = yf.download("KWEB", period=f"{lookback_days}d",
                            interval="1d", progress=False)["Close"]

        hstech_ret = hstech.pct_change().dropna()
        kweb_ret = kweb.pct_change().dropna()

        aligned = hstech_ret.align(kweb_ret, join="inner")
        h_ret, k_ret = aligned[0].values, aligned[1].values
        if len(h_ret) < 20:
            return None

        beta = np.cov(h_ret, k_ret)[0, 1] / np.var(k_ret)
        latest_kweb_ret = float(kweb_ret.iloc[-1])
        implied_change_pct = round(beta * latest_kweb_ret * 100, 3)

        direction = "上升" if implied_change_pct > 0.1 else (
            "下跌" if implied_change_pct < -0.1 else "持平")

        return {
            "implied_change_pct": implied_change_pct,
            "direction": direction,
            "method": "KWEB夜盤報酬 x 歷史Beta係數回歸(代理估算, 非HKEX官方COP)",
            "disclaimer": "此為模擬推算值，非官方真實期指開盤數據",
        }
    except Exception as e:
        print(f"警告: 無法估算恒指科技期指隱含開盤 ({e})")
        return None


def estimate_southbound_proxy(lookback_days=60):
    """
    當無法取得真實南向資金數據時，用市場公開數據模擬一個
    「疑似南向資金流向」代理分數（方向性指標，非真實金額）。

    組成訊號：
    1. AH股溢價變化（溢價收窄 = 偏買入H股訊號）
    2. 恒生ETF(2800.HK)量能異常（放量上漲 = 偏流入訊號）
    3. 國企指數(^HSCE)相對上證指數(000001.SS)超額報酬

    回傳: dict，包含 proxy_score(-1~1)、方向標籤、各分項z-score、免責聲明。
    失敗時回傳 None，交由prompt誠實標註「數據未提供」。
    """
    try:
        # 1. AH溢價代理：用恒生AH股溢價指數(^HSAHP)作為現成溢價指標
        ah_premium = yf.download("^HSAHP", period=f"{lookback_days}d",
                                  interval="1d", progress=False)["Close"]
        ah_change = ah_premium.pct_change().dropna()
        ah_signal = -ah_change.iloc[-1]  # 溢價下降(負變化) -> 轉正訊號(偏買H股)

        # 2. 2800.HK 量能異常
        etf = yf.download("2800.HK", period=f"{lookback_days}d",
                           interval="1d", progress=False)
        vol_ma20 = etf["Volume"].rolling(20).mean()
        vol_ratio = float(etf["Volume"].iloc[-1] / vol_ma20.iloc[-1]) - 1
        price_up = float(etf["Close"].pct_change().iloc[-1]) > 0
        etf_signal = vol_ratio if price_up else -vol_ratio

        # 3. 國企指數 vs 上證指數 相對超額報酬
        hscei = yf.download("^HSCE", period=f"{lookback_days}d",
                             interval="1d", progress=False)["Close"].pct_change()
        sse = yf.download("000001.SS", period=f"{lookback_days}d",
                           interval="1d", progress=False)["Close"].pct_change()
        aligned = hscei.align(sse, join="inner")
        excess_ret = (aligned[0] - aligned[1]).dropna()
        excess_signal = float(excess_ret.iloc[-1])

        # 標準化 (用各自序列的歷史std做簡易z-score)
        def z(signal, series_std):
            return signal / series_std if series_std > 0 else 0

        z1 = z(ah_signal, ah_change.std())
        z2 = z(etf_signal, 1.0)  # vol_ratio本身已是比例, 不再除std
        z3 = z(excess_signal, excess_ret.std())

        composite = round(float(np.clip((z1 + z2 + z3) / 3, -1, 1)), 3)
        direction = "疑似流入" if composite > 0.15 else (
            "疑似流出" if composite < -0.15 else "中性/不明顯")

        return {
            "proxy_score": composite,
            "direction": direction,
            "components": {
                "AH溢價訊號": round(float(z1), 3),
                "2800.HK量能訊號": round(float(z2), 3),
                "國企vs上證超額報酬": round(float(z3), 3),
            },
            "disclaimer": "此為市場公開數據推算之方向性代理指標，非HKEX官方南向資金真實淨流入金額，僅供參考",
        }
    except Exception as e:
        print(f"警告: 無法計算南向資金代理指標 ({e})")
        return None
