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
