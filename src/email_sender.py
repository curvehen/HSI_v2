import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from config import (EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVERS,
                     EMAIL_SUBJECT_PREFIX)


def _build_html_body(output):
    """
    將predict_watchlist.py嘅完整output(dict)轉做HTML格式email內容。
    包含: 總結表格(summary) + 每隻股票嘅詳細分析(llm_analysis)。
    """
    run_date = output.get("run_date", "")
    summary = output.get("summary", "") or "(本次無法生成總結)"
    results = output.get("results", [])

    # ---- 將Markdown表格粗略轉做HTML(summary本身由LLM輸出Markdown) ----
    summary_html = summary.replace("\n", "<br>")

    rows_html = ""
    for r in results:
        target = r.get("target", "未知")
        ticker = r.get("ticker", "")

        if r.get("error"):
            rows_html += f"""
            <div style="border:1px solid #ddd; margin-bottom:16px; padding:12px; border-radius:6px; background:#fff3f3;">
                <h3 style="margin:0 0 8px 0; color:#c0392b;">{target} ({ticker}) — 處理失敗</h3>
                <p style="margin:0; color:#555;">錯誤: {r.get("error")}</p>
            </div>
            """
            continue

        proba = r.get("ensemble_proba_up", 0)
        agreement = r.get("agreement", "")
        filter_pass = r.get("confidence_filter_pass", False)
        entry = r.get("suggested_entry")
        stop = r.get("suggested_stop_loss")
        target_price = r.get("suggested_take_profit")
        ohlc = r.get("predicted_ohlc", {})
        llm_text = (r.get("llm_analysis") or "(暫無LLM分析)").replace("\n", "<br>")

        direction_color = "#27ae60" if proba > 0.55 else ("#c0392b" if proba < 0.45 else "#7f8c8d")
        filter_badge = "✅ 通過" if filter_pass else "⚠️ 未通過(訊號矛盾或信心不足)"

        rows_html += f"""
        <div style="border:1px solid #ddd; margin-bottom:16px; padding:12px; border-radius:6px;">
            <h3 style="margin:0 0 8px 0;">{target} ({ticker})</h3>
            <p style="margin:4px 0;">
                <b>上升機率:</b> <span style="color:{direction_color}; font-weight:bold;">{proba:.2%}</span>
                &nbsp;|&nbsp; <b>模型對齊:</b> {agreement}
                &nbsp;|&nbsp; <b>信心過濾:</b> {filter_badge}
            </p>
            <p style="margin:4px 0;">
                <b>預測OHLC:</b> 開{ohlc.get('expected_open')} / 高{ohlc.get('expected_high')} /
                低{ohlc.get('expected_low')} / 收{ohlc.get('expected_close')}
            </p>
            <p style="margin:4px 0;">
                <b>建議入場:</b> {entry} &nbsp;|&nbsp; <b>止蝕:</b> {stop} &nbsp;|&nbsp; <b>止賺:</b> {target_price}
            </p>
            <details style="margin-top:8px;">
                <summary style="cursor:pointer; color:#2980b9;">查看LLM詳細分析</summary>
                <p style="font-size:13px; color:#333; margin-top:8px;">{llm_text}</p>
            </details>
        </div>
        """

    html = f"""
    <html>
    <body style="font-family: Arial, 'Microsoft JhengHei', sans-serif; color:#222; max-width:800px; margin:auto;">
        <h2 style="color:#2c3e50;">港股量化模型每日預測報告</h2>
        <p style="color:#888;">報告日期: {run_date} | 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div style="background:#f7f9fa; border-left:4px solid #2980b9; padding:12px; margin-bottom:20px;">
            <h3 style="margin-top:0;">今日總結</h3>
            <p>{summary_html}</p>
        </div>

        <h3>個股詳細預測</h3>
        {rows_html}

        <hr style="margin-top:30px;">
        <p style="font-size:12px; color:#999;">
            本報告由量化模型自動生成, 僅供參考, 不構成投資建議。
            所有預測基於歷史數據同統計模型, 存在不確定性及虧損風險, 請自行審慎判斷。
        </p>
    </body>
    </html>
    """
    return html


def send_prediction_email(output):
    """
    發送predict_watchlist.py嘅結果俾多個收件人。
    需要環境變數: EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD
    """
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
