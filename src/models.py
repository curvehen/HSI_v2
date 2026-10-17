import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y_train)
    return model, scaler

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lgb(X_train, y_train):
    model = lgb.LGBMClassifier(
        max_depth=4, n_estimators=200, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    return model

def predict_proba_logistic(model, scaler, X):
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]

def predict_proba_tree(model, X):
    return model.predict_proba(X)[:, 1]
