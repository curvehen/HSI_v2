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
        hstech = yf.download("3032.HK", period=f"{lookback_days}d", interval="1d", progress=False)["Close"]
        kweb = yf.download("KWEB", period=f"{lookback_days}d", interval="1d", progress=False)["Close"]

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
    ^HSAHP在Yahoo Finance無數據源, 故移除AH溢價訊號,
    改用「2800.HK量能」+「國企指數vs上證超額報酬」兩訊號合成代理分數。
    """
    try:
        # 1. 2800.HK 量能異常
        etf = yf.download("2800.HK", period=f"{lookback_days}d",
                           interval="1d", progress=False)
        vol_ma20 = etf["Volume"].rolling(20).mean()
        vol_ratio = float(etf["Volume"].iloc[-1] / vol_ma20.iloc[-1]) - 1
        price_up = float(etf["Close"].pct_change().iloc[-1]) > 0
        etf_signal = vol_ratio if price_up else -vol_ratio

        # 2. 國企指數 vs 上證指數 相對超額報酬
        hscei = yf.download("^HSCE", period=f"{lookback_days}d",
                             interval="1d", progress=False)["Close"].pct_change()
        sse = yf.download("000001.SS", period=f"{lookback_days}d",
                           interval="1d", progress=False)["Close"].pct_change()
        aligned = hscei.align(sse, join="inner")
        excess_ret = (aligned[0] - aligned[1]).dropna()
        excess_signal = float(excess_ret.iloc[-1])

        z2 = etf_signal
        z3 = excess_signal / excess_ret.std() if excess_ret.std() > 0 else 0

        composite = round(float(np.clip((z2 + z3) / 2, -1, 1)), 3)  # 只剩2個訊號平均
        direction = "疑似流入" if composite > 0.15 else (
            "疑似流出" if composite < -0.15 else "中性/不明顯")

        return {
            "proxy_score": composite,
            "direction": direction,
            "components": {
                "2800.HK量能訊號": round(float(z2), 3),
                "國企vs上證超額報酬": round(float(z3), 3),
            },
            "disclaimer": "此為市場公開數據推算之方向性代理指標(已移除AH溢價訊號, "
                          "因^HSAHP在Yahoo Finance無數據源)，非HKEX官方南向資金真實淨流入金額，僅供參考",
        }
    except Exception as e:
        print(f"警告: 無法計算南向資金代理指標 ({e})")
        return None

