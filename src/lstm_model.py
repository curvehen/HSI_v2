import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

SEQ_LEN = 20

def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, n_features, epochs=30):
    Xs, ys = make_sequences(X_train, y_train)
    if len(Xs) < 50:
        return None  # 數據不足, 跳過LSTM訓練

    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(Xs, ys, epochs=epochs, batch_size=16, verbose=0)
    return model

def predict_lstm(model, X_recent_seq):
    """X_recent_seq shape: (1, SEQ_LEN, n_features)"""
    if model is None:
        return None
    return float(model.predict(X_recent_seq, verbose=0)[0, 0])
