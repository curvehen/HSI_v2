import yfinance as yf


def fetch_latest_news(ticker, max_items=3):
    """
    用yfinance攞返呢隻股票最新嘅新聞標題(真實數據, 唔係虛構)。
    失敗或者冇新聞時回傳空list, 等LLM誠實咁答"冇重大消息"。
    """
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if not news:
            return []

        headlines = []
        for item in news[:max_items]:
            content = item.get("content", item)  # 兼容唔同yfinance版本結構
            title = content.get("title") or item.get("title")
            publisher = (content.get("provider", {}) or {}).get("displayName") \
                if isinstance(content.get("provider"), dict) else item.get("publisher")
            pub_date = content.get("pubDate") or item.get("providerPublishTime")

            if title:
                headlines.append({
                    "title": title,
                    "publisher": publisher or "未知來源",
                    "published": str(pub_date) if pub_date else "未知時間",
                })

        return headlines

    except Exception as e:
        print(f"警告: 無法下載 {ticker} 新聞 ({e})")
        return []

CATALYST_KEYWORDS = ["業績", "盈警", "併購", "收購", "政策", "制裁",
                      "降息", "加息", "財報", "停牌", "退市", "監管"]

def flag_potential_catalyst(headlines):
    """
    對news_headlines做簡易關鍵字掃描, 標記是否存在潛在催化消息。
    這只是輔助標記(非權威判斷), 最終解讀仍交由LLM從標題內容自行歸納。
    無標題或無關鍵字命中時回傳None, 避免誤導。
    """
    if not headlines:
        return None
    matched = [h["title"] for h in headlines
               if any(kw in h["title"] for kw in CATALYST_KEYWORDS)]
    if not matched:
        return None
    return {"potential_catalyst_titles": matched, "method": "關鍵字初篩, 非權威判斷"}
