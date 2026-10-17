import numpy as np
import pandas as pd
from config import TRANSACTION_COST_BPS

def run_backtest(df, proba_col="ensemble_proba", threshold=0.55, target_col="target_ret"):
    """
    df 必須有 ensemble_proba 同 target_ret 欄
    只喺機率超過threshold(long)或低於1-threshold(short)先交易
    """
    df = df.copy()
    df["direction"] = np.where(df[proba_col] > threshold, 1,
                        np.where(df[proba_col] < (1 - threshold), -1, 0))

    cost = TRANSACTION_COST_BPS / 10000.0
    df["trade_ret"] = df["direction"] * df[target_col] - (df["direction"].abs() * cost)

    traded = df[df["direction"] != 0]
    return df, traded

def compute_metrics(traded):
    n_trades = len(traded)
    if n_trades == 0:
        return {"n_trades": 0, "win_rate": None, "sharpe": None,
                "max_drawdown": None, "total_return": None}

    wins = (traded["trade_ret"] > 0).sum()
    win_rate = wins / n_trades

    mean_ret = traded["trade_ret"].mean()
    std_ret = traded["trade_ret"].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

    cum_ret = (1 + traded["trade_ret"]).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "n_trades": int(n_trades),
        "wins": int(wins),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else None,
        "max_drawdown": round(max_dd, 4),
        "total_return": round(cum_ret.iloc[-1] - 1, 4),
    }
