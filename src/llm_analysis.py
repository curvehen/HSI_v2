# AFTER
import os
import time
import requests
from config import OPENROUTER_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.llm_prompts import build_hsi_prompt, build_stock_prompt, build_watchlist_summary_prompt


def _call_llm(prompt_text, max_tokens=None, retries=2):
    """
    呼叫OpenRouter嘅DeepSeek-R1模型(think mode / reasoner)。
    R1會先輸出內部推理鏈(reasoning), 再輸出最終答案, 唔需要/no_think控制字串
    (呢個字串只係Qwen3專用指令, 對DeepSeek無效, 必須移除)。
    加入重試機制, 應對偶發性空回應。
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        msg = "環境變數OPENROUTER_API_KEY未設定"
        print(f"警告: {msg}")
        return f"[LLM分析暫時無法生成: {msg}]"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-project-domain.example",  # ⚠️ OpenRouter建議附上, 改成你實際網域/repo
        "X-Title": "HSI Quant Prediction",                        # ⚠️ 自訂, 顯示喺OpenRouter dashboard
    }

    tokens_to_use = max_tokens or LLM_MAX_TOKENS

    payload = {
        "model": OPENROUTER_MODEL,  # "deepseek/deepseek-r1"
        "messages": [
            {"role": "system", "content": "你是一位專業港股短線交易分析師, 只根據提供嘅數據分析, 絕不虛構數據。"},
            {"role": "user", "content": prompt_text},  # 唔再加/no_think, R1本身就要thinking
        ],
        "max_tokens": tokens_to_use,
        "temperature": LLM_TEMPERATURE,
    }

    last_error = None

    for attempt in range(1, retries + 2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)  # R1思考耗時更長, timeout拉長
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                print(f"警告: 第{attempt}次嘗試, OpenRouter回傳錯誤 ({result['error']})")
                last_error = str(result["error"])
                if attempt <= retries:
                    time.sleep(2)
                    continue
                break

            choices = result.get("choices")
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content")

                # DeepSeek R1透過OpenRouter有時會將思考鏈同答案一齊塞落content,
                # 用<think>...</think>包住, 保留原有剝除邏輯
                if text and text.strip():
                    if "<think>" in text and "</think>" in text:
                        text = text.split("</think>")[-1].strip()
                    return text
                else:
                    print(f"警告: 第{attempt}次嘗試, content為空 (max_tokens={tokens_to_use})")
                    last_error = "content為空(可能思考鏈耗盡max_tokens, 冇餘量輸出答案)"
                    if attempt <= retries:
                        time.sleep(2)
                        continue
            else:
                print(f"警告: 第{attempt}次嘗試, 回應冇choices欄位 ({result})")
                last_error = "冇choices欄位"
                if attempt <= retries:
                    time.sleep(2)
                    continue

        except requests.exceptions.RequestException as e:
            print(f"警告: 第{attempt}次嘗試, OpenRouter呼叫失敗 ({e})")
            last_error = str(e)
            if attempt <= retries:
                time.sleep(2)
                continue

    return f"[LLM分析為空或失敗, 已重試{retries}次仍未成功: {last_error}]"


def generate_hsi_analysis(payload):
    prompt = build_hsi_prompt(payload)
    return _call_llm(prompt, max_tokens=2500)


def generate_stock_analysis(payload):
    prompt = build_stock_prompt(payload)
    return _call_llm(prompt, max_tokens=2000)


def generate_watchlist_summary(results):
    prompt = build_watchlist_summary_prompt(results)
    # Summary要處理19隻股嘅合併表格, 需要大幅提高token上限
    return _call_llm(prompt, max_tokens=4000)
