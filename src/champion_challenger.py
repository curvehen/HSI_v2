import json
import os
import numpy as np
from src.features import FEATURE_COLUMNS
from src.hyperparam_search import load_champion_params
from src.stats_test import binomial_significance_test
from config import (CHAMPION_PARAMS_PATH, CHALLENGER_PARAMS_PATH,
                     CHALLENGER_MIN_IMPROVEMENT, CHALLENGER_MIN_TEST_TRADES)


def _backtest_params(df, params, test_ratio=0.3):
    import xgboost as xgb
    n = len(df)
    split = int(n * (1 - test_ratio))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_direction"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_direction"].values
    test_ret = test_df["target_ret"].values

    model = xgb.XGBClassifier(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    direction = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))
    trade_ret = direction * test_ret
    traded_mask = direction != 0
    n_trades = traded_mask.sum()

    if n_trades == 0:
        return {"n_trades": 0, "win_rate": None, "sharpe": None}

    traded_rets = trade_ret[traded_mask]
    win_rate = (traded_rets > 0).mean()
    sharpe = (traded_rets.mean() / traded_rets.std()) * np.sqrt(252) if traded_rets.std() > 0 else 0

    return {
        "n_trades": int(n_trades),
        "wins": int((traded_rets > 0).sum()),
        "win_rate": round(float(win_rate), 4),
        "sharpe": round(float(sharpe), 4),
    }


def run_champion_challenger_evaluation(df):
    """
    比較champion vs challenger喺同一段holdout數據嘅表現。
    只有當challenger顯著優於champion, 且樣本數足夠, 先更新champion。
    呢個function唔會自動部署, 只會回傳建議同比較結果, 最終部署需人手confirm。
    """
    champion_params = load_champion_params()

    if not os.path.exists(CHALLENGER_PARAMS_PATH):
        return {"status": "no_challenger", "message": "未有challenger參數, 請先run hyperparameter search"}

    with open(CHALLENGER_PARAMS_PATH, "r") as f:
        challenger_params = json.load(f)

    champion_result = _backtest_params(df, champion_params)
    challenger_result = _backtest_params(df, challenger_params)

    if challenger_result["n_trades"] < CHALLENGER_MIN_TEST_TRADES:
        return {
            "status": "insufficient_samples",
            "message": f"Challenger測試樣本不足({challenger_result['n_trades']}筆, 需要{CHALLENGER_MIN_TEST_TRADES}筆)",
            "champion_result": champion_result,
            "challenger_result": challenger_result,
        }

    champion_sharpe = champion_result["sharpe"] or -999
    challenger_sharpe = challenger_result["sharpe"] or -999
    improvement = challenger_sharpe - champion_sharpe

    should_promote = improvement >= CHALLENGER_MIN_IMPROVEMENT

    return {
        "status": "evaluated",
        "champion_result": champion_result,
        "challenger_result": challenger_result,
        "sharpe_improvement": round(improvement, 4),
        "should_promote": should_promote,
        "recommendation": (
            f"建議{'升級challenger為新champion' if should_promote else '維持現有champion'} "
            f"(sharpe差異: {improvement:+.4f}, 門檻: {CHALLENGER_MIN_IMPROVEMENT})"
        )
    }


def promote_challenger_to_champion():
    """
    人手confirm後執行: 將challenger參數正式變成champion。
    呢個function應該喺run_champion_challenger_evaluation()建議promote, 並經人手覆核後先調用。
    """
    if not os.path.exists(CHALLENGER_PARAMS_PATH):
        raise FileNotFoundError("冇challenger參數檔案")

    with open(CHALLENGER_PARAMS_PATH, "r") as f:
        challenger_params = json.load(f)

    os.makedirs(os.path.dirname(CHAMPION_PARAMS_PATH), exist_ok=True)

    # 保留舊champion做備份, 方便回滾
    if os.path.exists(CHAMPION_PARAMS_PATH):
        with open(CHAMPION_PARAMS_PATH, "r") as f:
            old_champion = json.load(f)
        backup_path = CHAMPION_PARAMS_PATH.replace(".json", "_backup.json")
        with open(backup_path, "w") as f:
            json.dump(old_champion, f, indent=2)
        print(f"舊champion已備份至 {backup_path}")

    with open(CHAMPION_PARAMS_PATH, "w") as f:
        json.dump(challenger_params, f, indent=2)

    print(f"Challenger已正式升級為新champion: {challenger_params}")
    return challenger_params


def rollback_to_backup():
    """
    緊急回滾: 若新champion部署後表現持續轉差, 可以用呢個function恢復返備份版本。
    """
    backup_path = CHAMPION_PARAMS_PATH.replace(".json", "_backup.json")
    if not os.path.exists(backup_path):
        raise FileNotFoundError("冇備份檔案可供回滾")

    with open(backup_path, "r") as f:
        backup_params = json.load(f)

    with open(CHAMPION_PARAMS_PATH, "w") as f:
        json.dump(backup_params, f, indent=2)

    print(f"已回滾至備份champion參數: {backup_params}")
    return backup_params
