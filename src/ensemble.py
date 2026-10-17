import numpy as np
from sklearn.linear_model import LogisticRegression

def stack_predictions(proba_dict, meta_model=None, y_train=None, train_mode=False):
    """
    proba_dict: {'xgb': array, 'lgb': array, 'logreg': array}
    若train_mode=True, 用out-of-fold機率訓練meta-learner
    否則直接用已訓練嘅meta_model做stacking預測
    """
    stacked_X = np.column_stack(list(proba_dict.values()))

    if train_mode:
        meta = LogisticRegression()
        meta.fit(stacked_X, y_train)
        return meta
    else:
        return meta_model.predict_proba(stacked_X)[:, 1]

def agreement_score(ensemble_proba, lstm_proba):
    """
    比較主力ensemble同LSTM獨立模型嘅方向一致性
    回傳: 'agree_strong' / 'agree_weak' / 'disagree'
    """
    if lstm_proba is None:
        return "no_lstm"

    ens_dir = 1 if ensemble_proba > 0.5 else 0
    lstm_dir = 1 if lstm_proba > 0.5 else 0

    if ens_dir == lstm_dir:
        conf_gap = abs(ensemble_proba - 0.5) + abs(lstm_proba - 0.5)
        return "agree_strong" if conf_gap > 0.3 else "agree_weak"
    else:
        return "disagree"
