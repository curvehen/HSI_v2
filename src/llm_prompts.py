import json


def build_hsi_prompt(payload):
    """
    HSI專用prompt, 根據你提供嘅完整格式。
    payload入面缺失嘅數據(如南向資金、期指隱含開盤)會係null,
    prompt明確要求LLM對null欄位回答"數據未提供", 禁止虛構。
    """
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = (
        "以下係一個量化模型對恆生指數下一個交易日嘅預測數據(JSON格式)。\n"
        "請用繁體中文、以專業但淺白嘅語氣,寫一段市場分析。\n\n"
        "規則:\n"
        "- 只根據JSON入面可取得的事實進行分析, 若某個欄位值為null或冇提供, "
        "必須明確講'數據未提供', 絕對不要虛構或假設任何即時數據(例如南向資金、期指隱含開盤、消息面, "
        "如JSON冇提供呢啲數據就直接講未有相關數據, 唔可以自行估算)\n"
        "- 內容短、狠、準, 不要寫成研究報告咁長篇大論\n"
        "- 事實與推論必須分開標註, 推論部分要註明'推論:'字眼\n\n"
        "請按以下格式輸出:\n\n"
        "【市場氛圍摘要】(100-150字, 包含市場氛圍判斷、主要支持訊號嘅數據點、一句風險提示)\n\n"
        "美股隔夜:\n"
        "南向資金:\n"
        "人民幣匯率:\n"
        "權重股動態:\n"
        "重要消息催化:\n"
        "恒指期貨隱含開盤方向:\n\n"
        "明日偏向: 上升 / 下跌 / 震盪\n"
        "概率: 上升__% / 下跌__% / 震盪__%\n"
        "信心: 高 / 中 / 低\n\n"
        "核心原因(按權重排序, 3-5點, 每點註明屬「基本面/資金/技術/情緒/外圍」):\n\n"
        "關鍵價位:\n"
        "支持1 / 支持2:\n"
        "阻力1 / 阻力2:\n"
        "ATR(14):\n"
        "期指高/低水啟示:\n\n"
        "交易情境:\n"
        "A. 造好情境:\n"
        "入場條件:\n入場位:\n止蝕位:\n目標位:\n風險回報比:\n\n"
        "B. 造淡情境:\n"
        "入場條件:\n入場位:\n止蝕位:\n目標位:\n風險回報比:\n\n"
        "C. 觀望情境:\n\n"
        "D. 下一交易日最可能情境與機率:\n\n"
        "E. 判斷可信度:\n\n"
        "3個最重要的驅動因素, 3個最重要的風險:\n\n"
        "咩情況下唔應該入場:\n\n"
        "失效條件(出現咩數據/走勢會推翻上面預測):\n\n"
        "明日風險事件(數據、業績、政策、外圍事件, 若JSON未提供請講'未有相關數據'):\n\n"
        "一句總結:\n\n"
        f"數據: {payload_str}"
    )
    return prompt


def build_stock_prompt(payload):
    """
    個股/重磅股/watchlist專用prompt, 根據你提供嘅格式。
    """
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = (
        "以下係一隻港股嘅模型預測數據(JSON格式)。你是專業短線交易員。\n"
        "請分析呢隻股票下一交易日走勢。\n\n"
        "規則:\n"
        "- 只根據JSON入面可取得的事實進行分析, 若某欄位為null或冇提供, 明確講'數據未提供', 不要虛構\n"
        "- 內容短、狠、準\n"
        "- 事實與推論分開, 推論部分註明'推論:'\n\n"
        "請根據技術面、量價、均線、MACD、RSI、KD、籌碼面、消息面、大盤與族群強弱, "
        "判斷明日最可能方向(若JSON冇提供某項數據, 該項請直接講'數據未提供', 不要憑空分析)。\n\n"
        "請直接輸出:\n\n"
        "1. 明日最可能走勢一句話結論:\n\n"
        "2. 上漲 / 震盪 / 下跌機率:\n\n"
        "3. 關鍵支撐與壓力位:\n\n"
        "4. 看多成立條件:\n\n"
        "5. 看空成立條件:\n\n"
        "6. 開盤後最重要的3個觀察訊號:\n\n"
        "7. 若做隔日短線, 較合理的交易思路:\n\n"
        "8. 本次判斷最不確定的因素:\n\n"
        f"數據: {payload_str}"
    )
    return prompt


def build_watchlist_summary_prompt(results):
    """
    將全部watchlist結果彙總做一個prompt, 要求LLM輸出一個合併總結表格。
    規則: 只有真實news_headlines先可以寫"原因", 冇新聞就寫"技術面/量化訊號, 無重大消息"。
    """
    import json

    simplified = []
    for r in results:
        if r.get("error"):
            continue
        simplified.append({
            "股票": r["target"],
            "代號": r["ticker"],
            "最新收市價": r.get("last_known_close"),
            "上升機率": r.get("ensemble_proba_up"),
            "模型對齊狀態": r.get("agreement"),
            "信心過濾結果": r.get("confidence_filter_pass"),
            "建議入場位": r.get("suggested_entry"),
            "止蝕位": r.get("suggested_stop_loss"),
            "止賺位": r.get("suggested_take_profit"),
            "RSI14": r.get("context_data", {}).get("RSI14"),
            "MACD_hist": r.get("context_data", {}).get("MACD_hist"),
            "真實新聞標題": r.get("context_data", {}).get("news_headlines", []),
        })

    payload_str = json.dumps(simplified, ensure_ascii=False)

    prompt = (
        "以下係一份港股watchlist嘅量化模型預測彙總數據(JSON格式), 包含多隻股票。\n\n"
        "請用繁體中文, 為每隻股票輸出一行總結, 組成一個表格。\n\n"
        "規則(必須嚴格遵守):\n"
        "- 「原因」一欄只可以根據JSON入面'真實新聞標題'呢個欄位嘅實際內容嚟寫, "
        "如果呢個list係空嘅, 你要睇下CCASS 有冇大變化, 有就出CCASS 的變化影響 否則, 必須寫'技術面/量化訊號, 無重大消息', 絕對不可以自行捏造任何新聞、"
        "業績、回購、公告等消息\n"
        "- 如果有真實新聞標題, 可以簡潔引用嗰個標題嘅重點, 但唔可以加工想像額外細節\n"
        "- 方向(買/賣/觀望): 根據AI 的計算出的結果 > 買; 賣; 觀望\n"
        "- 信心根據'根據AI 的計算出的結果: "
        "filter_pass=true且agree_strong -> 高; filter_pass=true且agree_weak -> 中; 其餘 -> 低\n\n"
        "請輸出一個Markdown表格, 欄位依次為: "
        "股票 | 代號 | 方向(買/賣/觀望) | 信心(高/中/低) | 建議入場位 | 止蝕位 | 原因\n\n"
        "表格之後, 加一段【總結】(50字以內), 指出今日watchlist入面邊幾隻訊號最強、邊幾隻要觀望。\n\n"
        f"數據: {payload_str}"
    )
    return prompt
