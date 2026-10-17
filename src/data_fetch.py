import yfinance as yf
import pandas as pd
from config import (HSI_TICKER, TOP_WEIGHT_TICKERS, MACRO_TICKERS, START_DATE,
                     AH_PAIRS, HSTECH_TICKER, USDCNH_TICKER)

def _dl(ticker, start):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ah_premium_proxy(start):
    """
    計算AH股溢價率變化, 作為南水情緒代理
    溢價率 = H股價格(換算做人民幣) / A股價格 - 1
    溢價率收窄(H股相對轉強) -> 可能反映南水買盤增加
    """
    usdcnh = _dl(USDCNH_TICKER, start)["Close"]  # 用嚟做港元/人民幣粗略換算參考

    premium_list = []
    for h_tk, a_tk in AH_PAIRS.items():
        try:
            h_price = _dl(h_tk, start)["Close"]
            a_price = _dl(a_tk, start)["Close"]
            # 簡化處理: 假設HKD兌CNY約為0.92 (實務應該用實時HKD/CNY匯率, 此處用近似值)
            hkd_cny_rate = 0.92
            h_price_cny = h_price * hkd_cny_rate
            premium = (h_price_cny / a_price - 1).rename(f"premium_{h_tk}")
            premium_list.append(premium)
        except Exception:
            continue

    if not premium_list:
        return pd.Series(dtype=float, name="ah_premium_avg")

    premium_df = pd.concat(premium_list, axis=1)
    ah_premium_avg = premium_df.mean(axis=1).rename("ah_premium_avg")
    return ah_premium_avg

def fetch_hstech_relative(start):
    """恒生科技 vs 恒指相對強弱, 做南水科網偏好代理"""
    try:
        hstech = _dl(HSTECH_TICKER, start)["Close"]
        hstech_ret = hstech.pct_change()
        return hstech_ret.rename("hstech_ret_raw")
    except Exception as e:
        print(f"警告: 無法下載{HSTECH_TICKER}數據 ({e}), 將用全零填充代替")
        return None


def fetch_usdcnh(start):
    """人民幣匯率變動, 做跨境資金流情緒代理"""
    usdcnh = _dl(USDCNH_TICKER, start)["Close"]
    usdcnh_ret = usdcnh.pct_change().rename("usdcnh_ret")
    return usdcnh_ret

def fetch_all(start=START_DATE):
    hsi = _dl(HSI_TICKER, start)[["Open", "High", "Low", "Close", "Volume"]].copy()
    hsi.columns = ["hsi_open", "hsi_high", "hsi_low", "hsi_close", "hsi_volume"]

    macro = {}
    for name, tk in MACRO_TICKERS.items():
        d = _dl(tk, start)["Close"]
        macro[name] = d
    macro_df = pd.DataFrame(macro)
    macro_df.columns = [f"{c}_close" for c in macro_df.columns]

    weight_closes = []
    weight_volumes = []
    for tk in TOP_WEIGHT_TICKERS:
        try:
            d = _dl(tk, start)
            weight_closes.append(d["Close"].pct_change().rename(tk))
            weight_volumes.append(d["Volume"].rename(tk))
        except Exception:
            continue
    weight_ret_df = pd.concat(weight_closes, axis=1)
    weight_vol_df = pd.concat(weight_volumes, axis=1)

    weight_summary = pd.DataFrame({
        "topweight_avg_ret": weight_ret_df.mean(axis=1),
        "topweight_avg_volume": weight_vol_df.mean(axis=1),
    })

    # 新增: 資金流代理指標
    print("計算AH溢價代理指標...")
    ah_premium = fetch_ah_premium_proxy(start)
    print("計算恒生科技相對強弱...")
    hstech_rel = fetch_hstech_relative(start)
    print("計算人民幣匯率變動...")
    usdcnh_ret = fetch_usdcnh(start)

    df = hsi.join(macro_df, how="inner")
    df = df.join(weight_summary, how="left")
    df = df.join(ah_premium, how="left")
    df = df.join(hstech_rel, how="left")
    df = df.join(usdcnh_ret, how="left")

    df.dropna(subset=["hsi_close"], inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df
