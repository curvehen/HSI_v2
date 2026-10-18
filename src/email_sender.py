import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from config import (EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVERS,
                     EMAIL_SUBJECT_PREFIX,SECTOR_MAP)


# ---------------------------------------------------------------------------
# 輔助函數: 將Model原始數值轉做人類可讀標籤
# ---------------------------------------------------------------------------

def _signal_label(proba):
    if proba is None:
        return "觀望", "#7f8c8d"
    if proba > 0.55:
        return "做多 ▲", "#27ae60"
    if proba < 0.45:
        return "做淡 ▼", "#c0392b"
    return "觀望 —", "#7f8c8d"


def _strength_label(proba, agreement):
    """根據機率偏離0.5嘅幅度 + LSTM對齊狀態, 判斷訊號強度(純量化規則, 非LLM主觀判斷)"""
    if proba is None:
        return "—"
    distance = abs(proba - 0.5)
    if agreement == "disagree":
        return "弱(分歧)"
    if distance > 0.35:
        return "強"
    if distance > 0.15:
        return "中"
    return "弱"


def _risk_reward_ratio(entry, stop, target):
    try:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0:
            return "N/A"
        return f"{reward / risk:.2f}x"
    except (TypeError, ZeroDivisionError):
        return "N/A"


def _extract_llm_short(llm_text, max_len=90):
    """
    從完整LLM分析文字中, 擷取最核心一句(供表格欄位用), 完整版放喈附錄。
    優先搵"一句話結論"或"一句總結"呢類關鍵句, 搵唔到就取前段文字。
    """
    if not llm_text:
        return "(暫無分析)"

    patterns = [
        r"一句話結論[:：]\s*(.+)",
        r"一句總結[:：]\s*(.+)",
    ]
    for p in patterns:
        match = re.search(p, llm_text)
        if match:
            line = match.group(1).strip().split("\n")[0]
            return line[:max_len] + ("..." if len(line) > max_len else "")

    cleaned = llm_text.strip().replace("\n", " ")
    return cleaned[:max_len] + ("..." if len(cleaned) > max_len else "")


def _pct(value):
    try:
        return f"{value * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# HSI 獨立預測區塊 (Model自己嘅完整預測, 參考dashboard風格)
# ---------------------------------------------------------------------------

def _build_hsi_section(hsi_result):
    if hsi_result is None or hsi_result.get("error"):
        return "<p style='color:#c0392b;'>HSI預測數據不可用。</p>"

    proba = hsi_result.get("ensemble_proba_up", 0)
    agreement = hsi_result.get("agreement", "")
    signal, signal_color = _signal_label(proba)
    strength = _strength_label(proba, agreement)

    last_close = hsi_result.get("last_known_close")
    ohlc = hsi_result.get("predicted_ohlc", {})
    pred_high = ohlc.get("expected_high")
    pred_low = ohlc.get("expected_low")
    pred_range = round(pred_high - pred_low, 1) if pred_high and pred_low else None
    range_pct = _pct((pred_range / last_close)) if pred_range and last_close else "N/A"

    entry = hsi_result.get("suggested_entry")
    stop = hsi_result.get("suggested_stop_loss")
    target = hsi_result.get("suggested_take_profit")
    rr_ratio = _risk_reward_ratio(entry, stop, target)

    llm_text = hsi_result.get("llm_analysis") or "(暫無LLM分析)"
    llm_html = llm_text.replace("\n", "<br>")

    html = f"""
    <div style="border:1px solid #ddd; border-radius:8px; padding:16px; margin-bottom:24px;">
        <h2 style="margin:0 0 4px 0;">🏛 HSI 預測報告(Model量化輸出)</h2>
        <p style="color:#888; margin:0 0 16px 0; font-size:13px;">生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div style="background:{'#eafaf1' if proba and proba>0.5 else '#fdecea'}; border-radius:6px; padding:12px; margin-bottom:16px;">
            <h2 style="margin:0; color:{signal_color};">{signal}</h2>
            <table style="width:100%; margin-top:8px; font-size:14px;">
                <tr>
                    <td>P(升): <b>{proba:.4f} ({_pct(proba)})</b></td>
                    <td>P(跌): <b>{1-proba:.4f} ({_pct(1-proba)})</b></td>
                </tr>
                <tr>
                    <td>信號強度: <b>{strength}</b></td>
                    <td>模型對齊: <b>{agreement}</b></td>
                </tr>
            </table>
        </div>

        <table style="width:100%; border-collapse:collapse; margin-bottom:16px; font-size:14px;">
            <tr style="background:#eaf2fb;"><td style="padding:8px;">預測高位</td>
                <td style="padding:8px; text-align:right; color:#2980b9; font-weight:bold;">{pred_high}</td></tr>
            <tr><td style="padding:8px;">今日收市(基準)</td>
                <td style="padding:8px; text-align:right;">{last_close}</td></tr>
            <tr style="background:#fdecea;"><td style="padding:8px;">預測低位</td>
                <td style="padding:8px; text-align:right; color:#c0392b; font-weight:bold;">{pred_low}</td></tr>
            <tr><td style="padding:8px;">預測範圍</td>
                <td style="padding:8px; text-align:right;">{pred_range} 點 ({range_pct})</td></tr>
        </table>

        <table style="width:100%; border-collapse:collapse; margin-bottom:12px; font-size:14px;">
            <tr style="background:#eafaf1;"><td style="padding:8px;">止盈目標</td>
                <td style="padding:8px; text-align:right; color:#27ae60; font-weight:bold;">{target}</td></tr>
            <tr style="background:#fdecea;"><td style="padding:8px;">建議止損</td>
                <td style="padding:8px; text-align:right; color:#c0392b; font-weight:bold;">{stop}</td></tr>
            <tr><td style="padding:8px;">風險回報比</td>
                <td style="padding:8px; text-align:right;">{rr_ratio}</td></tr>
        </table>

        <div style="background:#f7f9fa; border-left:4px solid #2980b9; padding:12px; border-radius:4px;">
            <b>💡 LLM Analysis(文字解讀, 非額外數據來源):</b>
            <p style="font-size:13px; color:#333; margin-top:8px;">{llm_html}</p>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# 個股預測表格 (watchlist, 最後一欄 = LLM速評)
# ---------------------------------------------------------------------------

def _build_stocks_table(stock_results):
    rows_html = ""
    appendix_html = ""

    for r in stock_results:
        if r.get("error"):
            rows_html += f"""
            <tr style="background:#fdecea;">
                <td style="padding:6px;">{r.get('target')}</td>
                <td colspan="10" style="padding:6px; color:#c0392b;">處理失敗: {r.get('error')}</td>
            </tr>
            """
            continue

        target = r.get("target")
        ticker = r.get("ticker")
        sector = SECTOR_MAP.get(ticker, "—")
        proba = r.get("ensemble_proba_up", 0)
        agreement = r.get("agreement", "")
        signal, signal_color = _signal_label(proba)
        strength = _strength_label(proba, agreement)

        last_close = r.get("last_known_close")
        ohlc = r.get("predicted_ohlc", {})
        pred_high = ohlc.get("expected_high")
        pred_low = ohlc.get("expected_low")
        pred_close = ohlc.get("expected_close")
        ret_pct = _pct((pred_close - last_close) / last_close) if last_close else "N/A"
        rsi = r.get("context_data", {}).get("RSI14", "N/A")

        # 命中率需要per-stock歷史backtest, 現階段未開發, 誠實標示"—"而非虛構數字
        hit_rate = "—"

        llm_short = _extract_llm_short(r.get("llm_analysis"))

        rows_html += f"""
        <tr>
            <td style="padding:6px; font-weight:bold;">{target}<br><span style="font-size:11px; color:#888;">{ticker}</span></td>
            <td style="padding:6px; text-align:center;">{sector}</td>
            <td style="padding:6px; color:{signal_color}; font-weight:bold;">{signal}</td>
            <td style="padding:6px; text-align:center;">{_pct(proba)}</td>
            <td style="padding:6px; text-align:center;">{strength}</td>
            <td style="padding:6px; text-align:right;">{last_close}</td>
            <td style="padding:6px; text-align:right; color:#2980b9;">{pred_high}</td>
            <td style="padding:6px; text-align:right; color:#c0392b;">{pred_low}</td>
            <td style="padding:6px; text-align:right;">{pred_close}</td>
            <td style="padding:6px; text-align:center;">{ret_pct}</td>
            <td style="padding:6px; text-align:center;">{rsi}</td>
            <td style="padding:6px; text-align:center; color:#aaa;">{hit_rate}</td>
            <td style="padding:6px; font-size:12px; color:#555; max-width:220px;">{llm_short}</td>
        </tr>
        """

        full_llm = (r.get("llm_analysis") or "(暫無分析)").replace("\n", "<br>")
        appendix_html += f"""
        <div style="border:1px solid #eee; padding:10px; margin-bottom:10px; border-radius:4px;">
            <b>{target} ({ticker}) — 完整LLM分析</b>
            <p style="font-size:12px; color:#333; margin-top:6px;">{full_llm}</p>
        </div>
        """

    table_html = f"""
    <h3>📈 個股預測(Watchlist)</h3>
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:13px; min-width:1100px;">
        <thead>
            <tr style="background:#2c3e50; color:white;">
                <th style="padding:8px; text-align:left;">股票</th>
                <th style="padding:8px;">板塊</th>
                <th style="padding:8px;">信號</th>
                <th style="padding:8px;">P(升)</th>
                <th style="padding:8px;">強度</th>
                <th style="padding:8px;">昨收</th>
                <th style="padding:8px;">預測高位</th>
                <th style="padding:8px;">預測低位</th>
                <th style="padding:8px;">預測收市</th>
                <th style="padding:8px;">1日回報</th>
                <th style="padding:8px;">RSI</th>
                <th style="padding:8px;">命中率</th>
                <th style="padding:8px; text-align:left;">LLM速評</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    <p style="font-size:11px; color:#999; margin-top:6px;">
        註: "命中率"需逐股歷史backtest數據, 現版本未開發, 顯示為"—", 不會以虛構數字填充。
    </p>

    <h3 style="margin-top:24px;">📋 個股完整LLM分析(附錄)</h3>
    {appendix_html}
    """
    return table_html


# ---------------------------------------------------------------------------
# 主組裝函數
# ---------------------------------------------------------------------------

def _build_html_body(output):
    run_date = output.get("run_date", "")
    results = output.get("results", [])

    hsi_result = next((r for r in results if r.get("target") == "HSI"), None)
    stock_results = [r for r in results if r.get("target") != "HSI"]

    hsi_section = _build_hsi_section(hsi_result)
    stocks_section = _build_stocks_table(stock_results)

    html = f"""
    <html>
    <body style="font-family: Arial, 'Microsoft JhengHei', sans-serif; color:#222; max-width:1000px; margin:auto;">
        <h1 style="color:#2c3e50;">港股量化模型每日預測報告</h1>
        <p style="color:#888;">報告日期: {run_date}</p>

        {hsi_section}
        {stocks_section}

        <hr style="margin-top:30px;">
        <p style="font-size:12px; color:#999;">
            本報告由量化模型自動生成, 數值部分(機率/OHLC/ATR/RSI/入場價位)由Model計算,
            文字分析部分由LLM根據數值生成解讀, 兩者性質不同, 僅供參考, 不構成投資建議。
        </p>
    </body>
    </html>
    """
    return html


def send_prediction_email(output):
    sender_address = os.environ.get("EMAIL_SENDER_ADDRESS")
    sender_password = os.environ.get("EMAIL_SENDER_PASSWORD")

    if not sender_address or not sender_password:
        print("警告: 環境變數EMAIL_SENDER_ADDRESS或EMAIL_SENDER_PASSWORD未設定, 跳過發送email")
        return False

    if not EMAIL_RECEIVERS:
        print("警告: EMAIL_RECEIVERS清單為空, 跳過發送email")
        return False

    run_date = output.get("run_date", "")
    subject = f"{EMAIL_SUBJECT_PREFIX} - {run_date}"
    html_body = _build_html_body(output)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_address
    msg["To"] = ", ".join(EMAIL_RECEIVERS)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(sender_address, sender_password)
            server.sendmail(sender_address, EMAIL_RECEIVERS, msg.as_string())

        print(f"Email已成功發送俾 {len(EMAIL_RECEIVERS)} 位收件人: {EMAIL_RECEIVERS}")
        return True

    except Exception as e:
        print(f"警告: Email發送失敗 ({e})")
        return False

