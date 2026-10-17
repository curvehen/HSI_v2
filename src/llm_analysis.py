import os
import requests
from config import CF_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.llm_prompts import build_hsi_prompt, build_stock_prompt, build_watchlist_summary_prompt

def _call_llm(prompt_text):
    """
    呼叫Cloudflare Workers AI嘅Qwen3-30B-A3B-FP8模型。
    需要環境變數: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
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
    payload = {
        "messages": [
            {"role": "system", "content": "你是一位專業港股短線交易分析師, 只根據提供嘅數據分析, 絕不虛構數據。"},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            return result["result"]["response"]
        else:
            errors = result.get("errors")
            print(f"警告: Cloudflare AI回傳錯誤 ({errors})")
            return f"[LLM分析暫時無法生成: {errors}]"

    except requests.exceptions.RequestException as e:
        print(f"警告: Cloudflare AI呼叫失敗 ({e})")
        return f"[LLM分析暫時無法生成: {e}]"


def generate_hsi_analysis(payload):
    prompt = build_hsi_prompt(payload)
    return _call_llm(prompt)


def generate_stock_analysis(payload):
    prompt = build_stock_prompt(payload)
    return _call_llm(prompt)

def generate_watchlist_summary(results):
    prompt = build_watchlist_summary_prompt(results)
    return _call_llm(prompt)
