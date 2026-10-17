import json
import os
import optuna
import numpy as np
from src.features import FEATURE_COLUMNS
from src.models import train_xgb, predict_proba_tree
from config import (CHAMPION_PARAMS_PATH, CHALLENGER_PARAMS_PATH,
                     HYPEROPT_N_TRIALS, PARAM_CHANGE_MAX_PCT)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_PARAMS = {
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def load_champion_params():
    if os.path.exists(CHAMPION_PARAMS_PATH):
        with open(CHAMPION_PARAMS_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


def _constrain_param(new_val, old_val, max_pct=PARAM_CHANGE_MAX_PCT):
    """Guardrail: 限制單次參數變動幅度, 避免劇烈震盪"""
    lower = old_val * (1 - max_pct)
    upper = old_val * (1 + max_pct)
    return float(np.clip(new_val, lower, upper))


def _walk_forward_objective(trial, df, champion_params, n_splits=4):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    n = len(df)
    fold_size = n // (n_splits + 1)
    scores = []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = train_end + fold_size
        if test_end > n:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["target_direction"].values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df["target_direction"].values

        import xgboost as xgb
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
        pred = (proba > 0.5).astype(int)
        accuracy = (pred == y_test).mean()
        scores.append(accuracy)

    return np.mean(scores) if scores else 0.0


def run_hyperparameter_search(df):
    """
    用Optuna做walk-forward超參數搜索, 搜索結果受PARAM_CHANGE_MAX_PCT約束,
    唔會偏離champion參數太遠。回傳challenger_params並存檔, 唔會自動取代champion。
    """
    champion_params = load_champion_params()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _walk_forward_objective(trial, df, champion_params),
        n_trials=HYPEROPT_N_TRIALS,
        show_progress_bar=False
    )

    raw_best = study.best_params

    # ---- Guardrail: 約束新參數唔好偏離champion太遠 ----
    constrained_params = {
        "max_depth": int(round(_constrain_param(raw_best["max_depth"], champion_params["max_depth"]))),
        "n_estimators": int(round(_constrain_param(raw_best["n_estimators"], champion_params["n_estimators"]))),
        "learning_rate": _constrain_param(raw_best["learning_rate"], champion_params["learning_rate"]),
        "subsample": _constrain_param(raw_best["subsample"], champion_params["subsample"]),
        "colsample_bytree": _constrain_param(raw_best["colsample_bytree"], champion_params["colsample_bytree"]),
    }

    os.makedirs(os.path.dirname(CHALLENGER_PARAMS_PATH), exist_ok=True)
    with open(CHALLENGER_PARAMS_PATH, "w") as f:
        json.dump(constrained_params, f, indent=2)

    print(f"Optuna搜索完成, best raw score: {study.best_value:.4f}")
    print(f"Champion params: {champion_params}")
    print(f"Challenger params(已約束): {constrained_params}")

    return constrained_params
