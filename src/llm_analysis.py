import os
import time
import requests
from config import CF_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.llm_prompts import build_hsi_prompt, build_stock_prompt, build_watchlist_summary_prompt


def _call_llm(prompt_text, max_tokens=None, retries=2):
    """
    呼叫Cloudflare Workers AI嘅Qwen3-30B-A3B-FP8模型。
    Qwen3係reasoning model, 用/no_think指令跳過內部推理, 直接輸出答案。
    加入重試機制, 應對偶發性空回應。
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        msg = "環境變數CLOUDFLARE_ACCOUNT_ID或CLOUDFLARE_API_TOKEN未設定"
        print(f"警告: {msg}")
        return f"[LLM分析暫時無法生成: {msg}]"

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CF_MODEL}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    tokens_to_use = max_tokens or LLM_MAX_TOKENS
    prompt_with_control = prompt_text + "\n\n/no_think"

    payload = {
        "messages": [
            {"role": "system", "content": "你係一位獨立嘅港股短線交易分析師, 只根據提供嘅原始市場數據分析, 絕不虛構數據, 亦都唔會參考或假設任何其他模型/系統嘅預測結論,你嘅判斷必須完全基於自己對數據嘅解讀。"},
            {"role": "user", "content": prompt_with_control},
        ],
        "max_tokens": tokens_to_use,
        "temperature": LLM_TEMPERATURE,
    }

    last_error = None

    for attempt in range(1, retries + 2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                text = result.get("result", {}).get("response")

                if text and text.strip():
                    if "<think>" in text and "</think>" in text:
                        text = text.split("</think>")[-1].strip()
                    return text
                else:
                    print(f"警告: 第{attempt}次嘗試, response為空 (max_tokens={tokens_to_use})")
                    last_error = "response為空"
                    if attempt <= retries:
                        time.sleep(2)
                        continue
            else:
                errors = result.get("errors")
                print(f"警告: 第{attempt}次嘗試, Cloudflare AI回傳錯誤 ({errors})")
                last_error = str(errors)
                if attempt <= retries:
                    time.sleep(2)
                    continue

        except requests.exceptions.RequestException as e:
            print(f"警告: 第{attempt}次嘗試, Cloudflare AI呼叫失敗 ({e})")
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
