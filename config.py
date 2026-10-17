HSI_TICKER = "^HSI"

TOP_WEIGHT_TICKERS = [
    "0700.HK", "9988.HK", "1299.HK", "0005.HK",
    "3690.HK", "0939.HK", "1398.HK", "2318.HK"
]

MACRO_TICKERS = {
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "dow": "^DJI",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "us10y": "^TNX",
}

AH_PAIRS = {
    "601398.SS": "1398.HK",
    "601939.SS": "0939.HK",
    "601318.SS": "2318.HK",
    "600036.SS": "3968.HK",
    "601288.SS": "1288.HK",
}

# 注意: Yahoo Finance無直接嘅恒生科技指數代碼(^HSTECH唔存在)
# 用3032.HK(恒生科技指數ETF)作為追蹤代理, 此ETF於2020年7月上市
HSTECH_TICKER = "3032.HK"
USDCNH_TICKER = "USDCNH=X"

START_DATE = "2015-01-01"

TRANSACTION_COST_BPS = 15
RISK_PER_TRADE = 0.01
ATR_STOP_MULT = 1.0
ATR_TARGET_MULT = 1.5

WALK_FORWARD_TRAIN_WINDOW = 500
WALK_FORWARD_TEST_WINDOW = 60
WALK_FORWARD_STEP = 60

SIM_START_DATE = "2026-01-01"
SIM_END_DATE = None
LSTM_RETRAIN_FREQ_DAYS = 5
TREE_RETRAIN_FREQ_DAYS = 1
MIN_TRAIN_HISTORY = 500

# ---- 自我學習機制設定 ----
PERFORMANCE_LOG_PATH = "predictions/performance_log.csv"
PREDICTION_HISTORY_PATH = "predictions/prediction_history.csv"
CHAMPION_PARAMS_PATH = "models_store/champion_params.json"
CHALLENGER_PARAMS_PATH = "models_store/challenger_params.json"

DRIFT_WINDOW = 30            # 滾動窗口: 最近30個預測
DRIFT_MIN_SAMPLES = 20       # 最少樣本數先做drift檢定
DRIFT_ACCURACY_FLOOR = 0.48  # 滾動準確度跌穿呢個水平 -> 觸發警報

HYPEROPT_N_TRIALS = 30       # Optuna搜索次數
HYPEROPT_MIN_HISTORY = 600   # 做hyperparameter search要最少幾多日數據
PARAM_CHANGE_MAX_PCT = 0.3   # 單次參數變動唔可以超過30% (guardrail)

CHALLENGER_MIN_IMPROVEMENT = 0.02  # challenger嘅sharpe要高過champion至少0.02先取代
CHALLENGER_MIN_TEST_TRADES = 50    # challenger驗證至少要有50筆交易先可信

MODEL_DIR = "models_store"
PRED_DIR = "predictions"


# ---- 預測目標設定 (可改做任何單一股票) ----
TARGET_TICKER = "^HSI"       # 改呢行: 例如 "0700.HK" (騰訊), "9988.HK" (阿里)
TARGET_LABEL = "HSI"         # 顯示用名稱, 跟住TARGET_TICKER一齊改, 例如 "Tencent"

# 大盤基準指數, 作為預測個股時嘅市場情緒特徵 (預測HSI自己時, 呢個會同TARGET重疊, 屬正常)
BENCHMARK_TICKER = "^HSI"

HSI_TICKER = "^HSI"  # 保留做BENCHMARK預設, 不要刪除

TOP_WEIGHT_TICKERS = [
    "0700.HK", "9988.HK", "1299.HK", "0005.HK",
    "3690.HK", "0939.HK", "1398.HK", "2318.HK"
]

MACRO_TICKERS = {
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "dow": "^DJI",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "us10y": "^TNX",
}

AH_PAIRS = {
    "601398.SS": "1398.HK",
    "601939.SS": "0939.HK",
    "601318.SS": "2318.HK",
    "600036.SS": "3968.HK",
    "601288.SS": "1288.HK",
}

HSTECH_TICKER = "3032.HK"
USDCNH_TICKER = "USDCNH=X"

START_DATE = "2015-01-01"

TRANSACTION_COST_BPS = 15
RISK_PER_TRADE = 0.01
ATR_STOP_MULT = 1.0
ATR_TARGET_MULT = 1.5

WALK_FORWARD_TRAIN_WINDOW = 500
WALK_FORWARD_TEST_WINDOW = 60
WALK_FORWARD_STEP = 60

SIM_START_DATE = "2026-01-01"
SIM_END_DATE = None
LSTM_RETRAIN_FREQ_DAYS = 5
TREE_RETRAIN_FREQ_DAYS = 1
MIN_TRAIN_HISTORY = 500

PERFORMANCE_LOG_PATH = "predictions/performance_log.csv"
PREDICTION_HISTORY_PATH = "predictions/prediction_history.csv"
CHAMPION_PARAMS_PATH = "models_store/champion_params.json"
CHALLENGER_PARAMS_PATH = "models_store/challenger_params.json"

DRIFT_WINDOW = 30
DRIFT_MIN_SAMPLES = 20
DRIFT_ACCURACY_FLOOR = 0.48

HYPEROPT_N_TRIALS = 30
HYPEROPT_MIN_HISTORY = 600
PARAM_CHANGE_MAX_PCT = 0.3

CHALLENGER_MIN_IMPROVEMENT = 0.02
CHALLENGER_MIN_TEST_TRADES = 50

MODEL_DIR = "models_store"
PRED_DIR = "predictions"

# ---- 批量預測清單: HSI + 重磅股 + 自訂Watchlist ----
WATCHLIST = {
    "HSI": "^HSI",
    "騰訊控股": "0700.HK",
    "阿里巴巴": "9988.HK",
    "中國移動": "0941.HK",
    "友邦保險": "1299.HK",
    "香港交易所": "0388.HK",
    "美團": "3690.HK",
    "滙豐控股": "0005.HK",
    "小米集團": "1810.HK",
    "中國平安": "2318.HK",
    "新鴻基地產": "0016.HK",
    "銀河娛樂": "0027.HK",
    "中國海洋石油": "0883.HK",
    "工商銀行": "1398.HK",
    "中國銀行": "3988.HK",
    "中電控股": "0002.HK",
    "長和": "0001.HK",
    " 比亞迪股份": "1211.HK",
    "信義光能": "0968.HK",
    # 可自行新增任何你想監察嘅股票, 格式: "顯示名稱": "ticker代碼"
}


# ---- LLM Analysis設定 (Cloudflare Workers AI) ----
# ---------- Cloudflare AI ----------
CF_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
LLM_MAX_TOKENS = 1500
LLM_TEMPERATURE = 0.4
ENABLE_LLM_ANALYSIS = True



