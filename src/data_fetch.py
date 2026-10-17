import yfinance as yf
import pandas as pd
import numpy as np
from config import (TARGET_TICKER, BENCHMARK_TICKER, TOP_WEIGHT_TICKERS,
                     MACRO_TICKERS, START_DATE, AH_PAIRS, HSTECH_TICKER, USDCNH_TICKER)


def _dl(ticker, start):
    """
    下載單一ticker嘅歷史數據, 並強制攤平MultiIndex columns。
    新版yfinance預設所有下載(包括單一ticker)都用MultiIndex columns,
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


def fetch_target_data(start, target_ticker):
    """下載預測目標嘅OHLCV, target_ticker由呼叫方傳入, 唔再寫死讀取config"""
    print(f"下載目標數據: {target_ticker}...")
    df = _dl(target_ticker, start)[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["target_open", "target_high", "target_low", "target_close", "target_volume"]
    return df



def fetch_benchmark_feature(start):
    """
    下載大盤基準(通常係HSI)作為輔助特徵, 用嚟判斷目標相對大盤嘅強弱。
    若TARGET本身就係BENCHMARK(即預測HSI自己), 呢個特徵會同target_ret_1d高度重疊,
    屬正常現象, 模型會自動學到呢個冗餘, 唔影響運作。
    """
    s = _safe_close_series(BENCHMARK_TICKER, start, label="benchmark")
    if s is None:
        return pd.Series(dtype=float, name="benchmark_ret")
    return s.pct_change().rename("benchmark_ret")


def fetch_macro_data(start):
    """下載所有宏觀/外圍市場數據, 單一ticker失敗唔影響其他"""
    macro = {}
    for name, tk in MACRO_TICKERS.items():
        s = _safe_close_series(tk, start, label=name)
        macro[name] = s if s is not None else pd.Series(dtype=float)

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
        print("警告: 所有重磅股下載失敗, topweight特徵將以預設值填充")
        return (pd.Series(dtype=float, name="topweight_avg_ret"),
                pd.Series(dtype=float, name="topweight_volume_ratio"))

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


def fetch_hstech_relative_proxy(start, benchmark_close_series):
    """計算恒生科技 vs 大盤基準嘅相對強弱代理指標"""
    try:
        hstech = _dl(HSTECH_TICKER, start)["Close"]
        hstech_ret = hstech.pct_change()

        if benchmark_close_series is None:
            print("警告: 基準數據不可用, tech_relative_strength_proxy將以0填充")
            return pd.Series(dtype=float, name="tech_relative_strength_proxy")

        bm_ret = benchmark_close_series.pct_change()
        combined = pd.DataFrame({"hstech": hstech_ret, "bm": bm_ret}).dropna()
        rel = (combined["hstech"] - combined["bm"]).rename("tech_relative_strength_proxy")
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


def fetch_all(start=START_DATE, target_ticker=None, benchmark_ticker=None):
    """
    主入口: 下載TARGET核心數據 + 全部輔助特徵。
    target_ticker/benchmark_ticker若唔傳入, 預設用config.py嘅TARGET_TICKER/BENCHMARK_TICKER,
    傳入時可覆蓋, 支援批量跑唔同股票。
    """
    tk = target_ticker or TARGET_TICKER
    bm = benchmark_ticker or BENCHMARK_TICKER

    target_df = fetch_target_data(start, tk)

    print("下載大盤基準數據...")
    bm_close = _safe_close_series(bm, start, label="benchmark")
    benchmark_feature = bm_close.pct_change().rename("benchmark_ret") if bm_close is not None \
        else pd.Series(dtype=float, name="benchmark_ret")

    print("下載宏觀/外圍市場數據...")
    macro_df = fetch_macro_data(start)

    print("下載重磅成分股數據...")
    topweight_avg_ret, topweight_volume_ratio = fetch_topweight_data(start)

    print("計算AH溢價代理指標...")
    ah_premium_change = fetch_ah_premium_proxy(start)

    print("計算恒生科技相對強弱代理指標...")
    tech_relative_strength = fetch_hstech_relative_proxy(start, bm_close)

    print("計算人民幣匯率代理指標...")
    usdcnh_ret = fetch_usdcnh_proxy(start)

    df = target_df.join(benchmark_feature, how="left")
    df = df.join(macro_df, how="left")
    df = df.join(topweight_avg_ret, how="left")
    df = df.join(topweight_volume_ratio, how="left")
    df = df.join(ah_premium_change, how="left")
    df = df.join(tech_relative_strength, how="left")
    df = df.join(usdcnh_ret, how="left")

    df.dropna(subset=["target_open", "target_high", "target_low", "target_close"], inplace=True)
    df.ffill(inplace=True)

    print(f"數據下載完成, 目標={tk}, 共 {len(df)} 個交易日, {len(df.columns)} 個原始欄位")
    return df
