from config import ATR_STOP_MULT, ATR_TARGET_MULT, RISK_PER_TRADE

def kelly_fraction(win_rate, win_loss_ratio):
    """半Kelly, 避免全Kelly過度波動"""
    f = win_rate - (1 - win_rate) / win_loss_ratio
    f = max(f, 0)
    return f * 0.5  # half-Kelly

def position_size(capital, win_rate, win_loss_ratio, risk_per_trade=RISK_PER_TRADE):
    kelly = kelly_fraction(win_rate, win_loss_ratio)
    sized_fraction = min(kelly, risk_per_trade * 5)  # 上限保護, 避免kelly估算爆走
    return capital * max(sized_fraction, risk_per_trade)

def compute_entry_exit(entry_price, atr, direction):
    """
    direction: 1 = long, 0 = short
    回傳 entry, stop_loss, take_profit
    """
    if direction == 1:
        stop_loss = entry_price - atr * ATR_STOP_MULT
        take_profit = entry_price + atr * ATR_TARGET_MULT
    else:
        stop_loss = entry_price + atr * ATR_STOP_MULT
        take_profit = entry_price - atr * ATR_TARGET_MULT
    return entry_price, stop_loss, take_profit

def predict_next_ohlc(last_close, overnight_gap_est, atr, direction_prob):
    """
    用統計法(非ML)推算次日OHLC區間, 避免對單一數值過度自信
    """
    expected_open = last_close * (1 + overnight_gap_est)

    if direction_prob > 0.5:
        bias = (direction_prob - 0.5) * 2  # 0~1, 信心越高bias越大
        expected_close = expected_open + atr * 0.3 * bias
        expected_high = expected_open + atr * (0.5 + 0.5 * bias)
        expected_low = expected_open - atr * (0.3 - 0.2 * bias)
    else:
        bias = (0.5 - direction_prob) * 2
        expected_close = expected_open - atr * 0.3 * bias
        expected_low = expected_open - atr * (0.5 + 0.5 * bias)
        expected_high = expected_open + atr * (0.3 - 0.2 * bias)

    return {
        "expected_open": round(expected_open, 1),
        "expected_high": round(expected_high, 1),
        "expected_low": round(expected_low, 1),
        "expected_close": round(expected_close, 1),
    }
