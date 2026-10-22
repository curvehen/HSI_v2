import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from config import (EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVERS,
                     EMAIL_SUBJECT_PREFIX, SECTOR_MAP)
from src.hit_rate import LOOKBACK_WINDOW, MIN_RECORDS_FOR_DISPLAY


# ---------------------------------------------------------------------------
# 輔助函數
# ---------------------------------------------------------------------------

def _signal_label(proba):
    if proba is None:
        return "觀望", "#7f8c8d"
    if proba > 0.55:
        return "做多 ▲", "#27ae60"
    if proba < 0.45:
        return "做淡 ▼", "#c0392b"
    return "觀望 —", "#7f8c8d"


def _strength_label(proba, agreement, lstm_proba=None):
    if proba is None:
        return "—"
    distance = abs(proba - 0.5)
    base = "強" if distance > 0.35 else "中" if distance > 0.15 else "弱"
    if agreement == "disagree":
        gap = abs(proba - lstm_proba) if lstm_proba is not None else None
        gap_str = f", 模型分歧度{gap:.2f}" if gap is not None else ""
        return f"{base}(分歧{gap_str})"
    return base




def _pct(value):
    try:
        return f"{value * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _signed_pct(part, base):
    """計算相對基準嘅帶正負號百分比, 對應圖入面 +0.98% / -0.98% 呢類顯示"""
    try:
        pct = (part - base) / base * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"
    except (TypeError, ZeroDivisionError):
        return "N/A"


def _extract_llm_short(llm_text, max_len=90):
    if not llm_text:
        return "(暫無分析)"
    patterns = [r"一句話結論[:：]\s*(.+)", r"一句總結[:：]\s*(.+)"]
    for p in patterns:
        match = re.search(p, llm_text)
        if match:
            line = match.group(1).strip().split("\n")[0]
            return line[:max_len] + ("..." if len(line) > max_len else "")
    cleaned = llm_text.strip().replace("\n", " ")
    return cleaned[:max_len] + ("..." if len(cleaned) > max_len else "")


def _regime_label(regime):
    mapping = {"LOW_VOL": "低波動", "NEUTRAL": "NEUTRAL", "HIGH_VOL": "高波動", "UNKNOWN": "—"}
    return mapping.get(regime, "—")


# ---------------------------------------------------------------------------
# HSI 獨立區塊 — Add LLM Analysis after 風險回報比
# ---------------------------------------------------------------------------

def _build_hsi_section(hsi_result):
    if hsi_result is None or hsi_result.get("error"):
        return "<p style='color:#c0392b;'>HSI預測數據不可用。</p>"

    proba = hsi_result.get("ensemble_proba_up", 0)
    agreement = hsi_result.get("agreement", "")
    signal, signal_color = _signal_label(proba)
    strength = _strength_label(proba, agreement)
    regime = _regime_label(hsi_result.get("regime"))

    last_close = hsi_result.get("last_known_close")
    ohlc = hsi_result.get("predicted_ohlc", {})
    pred_high = ohlc.get("expected_high")
    pred_low = ohlc.get("expected_low")
    pred_range = round(pred_high - pred_low, 1) if pred_high and pred_low else None

    high_pct = _signed_pct(pred_high, last_close)
    low_pct = _signed_pct(pred_low, last_close)
    range_pct = _pct(pred_range / last_close) if pred_range and last_close else "N/A"

    entry = hsi_result.get("suggested_entry")
    stop = hsi_result.get("suggested_stop_loss")
    target = hsi_result.get("suggested_take_profit")
    rr_ratio = hsi_result.get("risk_reward_ratio", "N/A")
    target_pct = _signed_pct(target, last_close)
    stop_pct = _signed_pct(stop, last_close)

    llm_text = hsi_result.get("llm_analysis") or "(暫無LLM分析)"
    llm_html = llm_text.replace("\n", "<br>")

    run_date = hsi_result.get("run_date", "")

    html = f"""
    <div style="border:1px solid #ddd; border-radius:8px; padding:16px; margin-bottom:24px;">
        <h2 style="margin:0 0 4px 0;">🏛 HSI 預測報告</h2>
        <p style="color:#888; margin:0 0 16px 0; font-size:13px;">
            預測日期: {run_date} | 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>

        <div style="background:{'#eafaf1' if proba and proba>0.5 else '#fdecea'}; border-radius:6px; padding:12px; margin-bottom:16px;">
            <h2 style="margin:0; color:{signal_color};">{signal}</h2>
            <table style="width:100%; margin-top:8px; font-size:14px;">
                <tr>
                    <td>P(升): <b>{proba:.4f} ({_pct(proba)})</b></td>
                    <td>P(跌): <b>{1-proba:.4f} ({_pct(1-proba)})</b></td>
                </tr>
                <tr>
                    <td>信號強度: <b>{strength}</b></td>
                    <td>Regime: <b>{regime}</b></td>
                </tr>
            </table>
        </div>

        <h4 style="margin-bottom:6px;">📊 恒指高低位預測</h4>
        <table style="width:100%; border-collapse:collapse; margin-bottom:16px; font-size:14px;">
            <tr style="background:#eaf2fb;">
                <td style="padding:8px;">預測高位</td>
                <td style="padding:8px; text-align:right; color:#2980b9; font-weight:bold;">{pred_high}</td>
                <td style="padding:8px; text-align:right; color:#2980b9;">{high_pct}</td>
            </tr>
            <tr>
                <td style="padding:8px;">今日收市</td>
                <td style="padding:8px; text-align:right;">{last_close}</td>
                <td style="padding:8px; text-align:right; color:#888;">基準</td>
            </tr>
            <tr style="background:#fdecea;">
                <td style="padding:8px;">預測低位</td>
                <td style="padding:8px; text-align:right; color:#c0392b; font-weight:bold;">{pred_low}</td>
                <td style="padding:8px; text-align:right; color:#c0392b;">{low_pct}</td>
            </tr>
            <tr>
                <td style="padding:8px;">預測範圍</td>
                <td style="padding:8px; text-align:right;">{pred_range} 點</td>
                <td style="padding:8px; text-align:right;">{range_pct}</td>
            </tr>
        </table>

        <h4 style="margin-bottom:6px;">💡 交易建議</h4>
        <table style="width:100%; border-collapse:collapse; margin-bottom:12px; font-size:14px;">
            <tr style="background:#eafaf1;">
                <td style="padding:8px;">止盈目標</td>
                <td style="padding:8px; text-align:right; color:#27ae60; font-weight:bold;">{target}</td>
                <td style="padding:8px; text-align:right; color:#27ae60;">{target_pct}</td>
            </tr>
            <tr style="background:#fdecea;">
                <td style="padding:8px;">建議止損</td>
                <td style="padding:8px; text-align:right; color:#c0392b; font-weight:bold;">{stop}</td>
                <td style="padding:8px; text-align:right; color:#c0392b;">{stop_pct}</td>
            </tr>
            <tr>
                <td style="padding:8px;">風險回報比</td>
                <td colspan="2" style="padding:8px; text-align:right;">{rr_ratio}</td>
            </tr>
        </table>

        <div style="background:#f7f9fa; border-left:4px solid #2980b9; padding:12px; border-radius:4px;">
            <b>🤖 LLM Analysis(文字解讀, 數值以上方Model輸出為準):</b>
            <p style="font-size:13px; color:#333; margin-top:8px; line-height:1.6;">{llm_html}</p>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# 個股預測表格 — Add LLM Analysis after 命中率(表格最後一欄)
# ---------------------------------------------------------------------------

def _build_stocks_table(stock_results):
    rows_html = ""
    appendix_html = ""

    for r in stock_results:
        if r.get("error"):
            rows_html += f"""
            <tr style="background:#fdecea;">
                <td style="padding:6px;">{r.get('target')}</td>
                <td colspan="11" style="padding:6px; color:#c0392b;">處理失敗: {r.get('error')}</td>
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
        ret_pct = _signed_pct(pred_close, last_close)
        rsi = r.get("RSI14", "N/A")
        if isinstance(rsi, float):
            rsi = f"{rsi:.1f}"

        hit_rate = r.get("hit_rate", "—")
        hit_rate_n = r.get("hit_rate_samples", 0)

        llm_short = _extract_llm_short(r.get("llm_analysis"))

        rows_html += f"""
        <tr>
            <td style="padding:6px; font-weight:bold;">{target}<br>
                <span style="font-size:11px; color:#888;">{ticker}</span></td>
            <td style="padding:6px; text-align:center;">{sector}</td>
            <td style="padding:6px; color:{signal_color}; font-weight:bold;">{signal}</td>
            <td style="padding:6px; text-align:center;">{_pct(proba)}</td>
            <td style="padding:6px; text-align:center;">{strength}</td>
            <td style="padding:6px; text-align:right;">{last_close}</td>
            <td style="padding:6px; text-align:right; color:#2980b9;">{pred_high}</td>
            <td style="padding:6px; text-align:right; color:#c0392b;">{pred_low}</td>
            <td style="padding:6px; text-align:right;">{pred_close}</td>
            <td style="padding:6px; text-align:center; color:{'#c0392b' if ret_pct.startswith('-') else '#27ae60'};">{ret_pct}</td>
            <td style="padding:6px; text-align:center;">{rsi}</td>
            <td style="padding:6px; text-align:center;" title="樣本數: {hit_rate_n}">{hit_rate}</td>
            <td style="padding:6px; font-size:12px; color:#555; max-width:220px;">{llm_short}</td>
        </tr>
        """

        full_llm = (r.get("llm_analysis") or "(暫無分析)").replace("\n", "<br>")
        appendix_html += f"""
        <div style="border:1px solid #eee; padding:10px; margin-bottom:10px; border-radius:4px;">
            <b>{target} ({ticker}) — 完整LLM分析</b>
            <p style="font-size:12px; color:#333; margin-top:6px; line-height:1.6;">{full_llm}</p>
        </div>
        """

    table_html = f"""
    <h3 style="margin-bottom:8px;">📈 個股預測</h3>
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:13px; min-width:1150px;">
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
        註: "命中率"為過去最多{LOOKBACK_WINDOW}次預測嘅方向準繩度, 樣本少於{MIN_RECORDS_FOR_DISPLAY}次時顯示"—", 不以虛構數字填充。
    </p>

    <h3 style="margin-top:24px;">📋 個股完整LLM分析(附錄)</h3>
    {appendix_html}
    """
    return table_html


# ---------------------------------------------------------------------------
# 主組裝 + 發送
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
    <body style="font-family: Arial, 'Microsoft JhengHei', sans-serif; color:#222; max-width:1200px; margin:auto;">
        <h1 style="color:#2c3e50;">港股量化模型每日預測報告</h1>
        <p style="color:#888;">報告日期: {run_date}</p>

        {hsi_section}
        {stocks_section}

        <hr style="margin-top:30px;">
        <p style="font-size:12px; color:#999;">
            本報告由量化模型自動生成, 數值部分(機率/OHLC/ATR/RSI/入場價位/命中率)由Model計算,
            文字分析部分由LLM根據以上數值生成解讀, 兩者性質不同, 僅供參考, 不構成投資建議。
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

# 新增輔助函數
def compare_model_vs_llm(r):
    """
    比對量化模型方向 vs LLM獨立判斷方向, 標記一致/分歧。
    注意: 這只是事後比對, LLM生成當下完全不知道模型結論。
    """
    model_direction = "上升" if r.get("ensemble_proba_up", 0.5) > 0.5 else "下跌"
    llm_text = r.get("llm_analysis", "")
    # 簡易關鍵字判斷LLM傾向(可依實際輸出格式調整比對邏輯)
    if "上升" in llm_text[:200]:
        llm_direction = "上升"
    elif "下跌" in llm_text[:200]:
        llm_direction = "下跌"
    else:
        llm_direction = "不明確"

    agree = (model_direction == llm_direction)
    return {
        "model_direction": model_direction,
        "llm_direction": llm_direction,
        "agree": agree,
        "label": "✅ 兩者一致" if agree else "⚠️ 兩者分歧(值得留意)",
    }
