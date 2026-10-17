import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 抑制TensorFlow INFO/WARNING log(CUDA提示等)

import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception as e:
    print(f"警告: TensorFlow載入失敗 ({e}), LSTM功能將停用")
    TF_AVAILABLE = False

SEQ_LEN = 20


def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, n_features, epochs=30):
    if not TF_AVAILABLE:
        return None

    Xs, ys = make_sequences(X_train, y_train)
    if len(Xs) < 50 or len(np.unique(ys)) < 2:
        return None

    try:
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
    except Exception as e:
        print(f"警告: LSTM訓練失敗 ({e}), 跳過LSTM對齊層")
        return None


def predict_lstm(model, X_recent_seq):
    if model is None:
        return None
    try:
        return float(model.predict(X_recent_seq, verbose=0)[0, 0])
    except Exception as e:
        print(f"警告: LSTM預測失敗 ({e})")
        return None
