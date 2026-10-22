# src/macro_data.py (新檔案)
"""
宏觀輔助數據抓取模組：南向資金、恒生科技期指隱含開盤估算。
這些數據並非HKEX官方即時API輸出，使用時須在LLM prompt/報告中註明來源與局限性。
"""
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
