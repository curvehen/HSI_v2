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
