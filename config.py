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

START_DATE = "2015-01-01"

TRANSACTION_COST_BPS = 15  # 手續費+印花稅+滑點 estimate, 15 basis points round-trip
RISK_PER_TRADE = 0.01      # 每單風險佔資金1%
ATR_STOP_MULT = 1.0
ATR_TARGET_MULT = 1.5

WALK_FORWARD_TRAIN_WINDOW = 500   # trading days
WALK_FORWARD_TEST_WINDOW = 60
WALK_FORWARD_STEP = 60

MODEL_DIR = "models_store"
PRED_DIR = "predictions"

# 加喺原本config.py後面

# AH股對照表 (H股代碼 -> 對應A股代碼)
AH_PAIRS = {
    "1398.HK": "601398.SS",  # 建設銀行
    "0939.HK": "601939.SS",  # 建設銀行 (另一代碼, 視乎實際可用性自行調整)
    "2318.HK": "601318.SS",  # 平安
    "0386.HK": "600028.SS",  # 中石化
    "0857.HK": "601857.SS",  # 中石油
}

HSTECH_TICKER = "3032.HK"  # 恒生科技指數ETF, 因yfinance無直接指數代碼^HSTECH

USDCNH_TICKER = "USDCNH=X"

SIM_START_DATE = "2026-01-01"
SIM_END_DATE = None  # None = 跑到最新可用日期
LSTM_RETRAIN_FREQ_DAYS = 5   # LSTM每5個交易日重訓一次
TREE_RETRAIN_FREQ_DAYS = 1   # 樹模型每日重訓
MIN_TRAIN_HISTORY = 500      # 最少要有幾多日歷史先開始模擬

