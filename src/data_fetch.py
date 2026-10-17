import yfinance as yf
import pandas as pd
import numpy as np
from config import (HSI_TICKER, TOP_WEIGHT_TICKERS, MACRO_TICKERS, START_DATE,
                     AH_PAIRS, HSTECH_TICKER, USDCNH_TICKER)


def _dl(ticker, start):
    """
    下載單一ticker嘅歷史數據, 並強制攤平MultiIndex columns。
    新版yfinance(2025年後)預設所有下載(包括單一ticker)都用MultiIndex columns,
    呢個function確保回傳嘅df永遠係單層columns, 避免下游運算出錯。
    """
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _safe_close_series(ticker, start, label=None):
    """
    安全攞取單一ticker嘅Close價Series, 失敗時回傳None並印警告,
    唔會raise exception中斷成個pipeline。
    """
    name = label or ticker
    try:
        df = _dl(ticker, start)
        if "Close" not in df.columns:
            print(f"警告: {name} 數據冇Close欄位, 跳過")
            return None
        return df["Close"]
    except Exception as e:
        print(f"警告: 無法下載 {name} 數據 ({e}), 將以中性值填充代替")
        return None


def fetch_macro_data(start):
    """下載所有宏觀/外圍市場數據, 單一ticker失敗唔影響其他"""
    macro = {}
    for name, tk in MACRO_TICKERS.items():
        s = _safe_close_series(tk, start, label=name)
        if s is not None:
            macro[name] = s
        else:
            macro[name] = pd.Series(dtype=float)

    macro_df = pd.DataFrame(macro)
    macro_df.columns = [f"{c}_close" for c in macro_df.columns]
    return macro_df


def fetch_topweight_data(start):
    """下載重磅成分股回報+成交量, 逐隻容錯"""
    weight_rets = []
    weight_vols = []

    for tk in TOP_WEIGHT_TICKERS:
        try:
            df = _dl(tk, start)
            weight_rets.append(df["Close"].pct_change().rename(tk))
            weight_vols.append(df["Volume"].rename(tk))
        except Exception as e:
            print(f"警告: 無法下載重磅股 {tk} 數據 ({e}), 跳過此股票")
            continue

    if len(weight_rets) == 0:
        print("警告: 所有重磅股下載失敗, topweight特徵將以0填充")
        return pd.Series(dtype=float, name="topweight_avg_ret"), \
               pd.Series(dtype=float, name="topweight_volume_ratio")

    weight_ret_df = pd.concat(weight_rets, axis=1)
    topweight_avg_ret = weight_ret_df.mean(axis=1).rename("topweight_avg_ret")

    weight_vol_df = pd.concat(weight_vols, axis=1)
    weight_vol_avg = weight_vol_df.mean(axis=1)
    weight_vol_ma20 = weight_vol_avg.rolling(20).mean()
    topweight_volume_ratio = (weight_vol_avg / weight_vol_ma20.replace(0, np.nan)) \
        .rename("topweight_volume_ratio")

    return topweight_avg_ret, topweight_volume_ratio


def fetch_ah_premium_proxy(start):
    """計算AH股溢價率變化代理指標, 單一配對失敗唔影響其他配對"""
    ah_premium_list = []
    for a_tk, h_tk in AH_PAIRS.items():
        try:
            a_price = _dl(a_tk, start)["Close"]
            h_price = _dl(h_tk, start)["Close"]
            pair_df = pd.DataFrame({"a": a_price, "h": h_price}).dropna()
            if len(pair_df) == 0:
                continue
            premium_ratio = pair_df["a"] / pair_df["h"]
            ah_premium_list.append(premium_ratio.rename(f"{a_tk}_premium"))
        except Exception as e:
            print(f"警告: AH配對 {a_tk}/{h_tk} 下載失敗 ({e}), 跳過")
            continue

    if len(ah_premium_list) == 0:
        print("警告: 所有AH配對下載失敗, ah_premium_change_proxy將以0填充")
        return pd.Series(dtype=float, name="ah_premium_change_proxy")

    ah_df = pd.concat(ah_premium_list, axis=1)
    ah_avg_premium = ah_df.mean(axis=1)
    return ah_avg_premium.pct_change().rename("ah_premium_change_proxy")


def fetch_hstech_relative_proxy(start, hsi_close_series):
    """計算恒生科技 vs 恒指相對強弱代理指標, 失敗時回傳None"""
    try:
        hstech = _dl(HSTECH_TICKER, start)["Close"]
        hstech_ret = hstech.pct_change()
        hsi_ret_tmp = hsi_close_series.pct_change()
        combined = pd.DataFrame({"hstech": hstech_ret, "hsi": hsi_ret_tmp}).dropna()
        rel = (combined["hstech"] - combined["hsi"]).rename("tech_relative_strength_proxy")
        return rel
    except Exception as e:
        print(f"警告: 無法下載 {HSTECH_TICKER} 數據 ({e}), tech_relative_strength_proxy將以0填充")
        return pd.Series(dtype=float, name="tech_relative_strength_proxy")


def fetch_usdcnh_proxy(start):
    """人民幣匯率變動代理指標"""
    s = _safe_close_series(USDCNH_TICKER, start, label="USDCNH")
    if s is None:
        return pd.Series(dtype=float, name="usdcnh_ret_proxy")
    return s.pct_change().rename("usdcnh_ret_proxy")


def fetch_all(start=START_DATE):
    """
    主入口: 下載HSI核心數據 + 所有特徵來源, 全面容錯。
    即使部分輔助數據源失敗, HSI核心數據唔會受影響, 代理特徵會以中性值填充。
    """
    print("下載HSI核心數據...")
    hsi_raw = _dl(HSI_TICKER, start)[["Open", "High", "Low", "Close", "Volume"]].copy()
    hsi_raw.columns = ["hsi_open", "hsi_high", "hsi_low", "hsi_close", "hsi_volume"]

    print("下載宏觀/外圍市場數據...")
    macro_df = fetch_macro_data(start)

    print("下載重磅成分股數據...")
    topweight_avg_ret, topweight_volume_ratio = fetch_topweight_data(start)

    print("計算AH溢價代理指標...")
    ah_premium_change = fetch_ah_premium_proxy(start)

    print("計算恒生科技相對強弱代理指標...")
    tech_relative_strength = fetch_hstech_relative_proxy(start, hsi_raw["hsi_close"])

    print("計算人民幣匯率代理指標...")
    usdcnh_ret = fetch_usdcnh_proxy(start)

    df = hsi_raw.join(macro_df, how="left")
    df = df.join(topweight_avg_ret, how="left")
    df = df.join(topweight_volume_ratio, how="left")
    df = df.join(ah_premium_change, how="left")
    df = df.join(tech_relative_strength, how="left")
    df = df.join(usdcnh_ret, how="left")

    # 核心HSI欄位必須存在, 其他輔助欄位容許缺失(後面build_features會fillna)
    df.dropna(subset=["hsi_open", "hsi_high", "hsi_low", "hsi_close"], inplace=True)
    df.ffill(inplace=True)

    print(f"數據下載完成, 共 {len(df)} 個交易日, {len(df.columns)} 個原始欄位")
    return df
