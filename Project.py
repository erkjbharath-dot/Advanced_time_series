"""
Advanced Time Series Forecasting with Deep Learning + Attention
Author: YOUR NAME
"""

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Concatenate
from tensorflow.keras.models import Model

# ==========================================
# 1. GENERATE MULTIVARIATE SYNTHETIC DATASET
# ==========================================
def generate_dataset(n_steps=1500):
    """
    Generates synthetic multivariate time series data with:
    - Trend
    - Weekly seasonality
    - Non-linear cross-feature dependencies
    """
    t = np.arange(n_steps)

    # Feature components
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50)  # 50-step cycle

    # Define 5 correlated features
    f1 = trend + seasonal + np.random.normal(0, 0.5, n_steps)
    f2 = 0.6 * f1 + np.random.normal(0, 0.3, n_steps)
    f3 = np.sin(f1) + np.random.normal(0, 0.2, n_steps)
    f4 = np.cos(f2) + np.random.normal(0, 0.2, n_steps)
    f5 = 0.3*f1 + 0.4*f2 + np.random.normal(0, 0.4, n_steps)

    df = pd.DataFrame({
        "feature_1": f1,
        "feature_2": f2,
        "feature_3": f3,
        "feature_4": f4,
        "feature_5": f5,
    })
    return df

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
def create_sequences(data, seq_len=30):
    """
    Converts time series to supervised learning format.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

# ==========================================
# 3. CUSTOM ATTENTION LAYER
# ==========================================
class Attention(Layer):
    """
    Standard additive attention mechanism.
    """
    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        score = tf.nn.softmax(tf.keras.layers.Dense(1)(inputs), axis=1)
        context = tf.reduce_sum(score * inputs, axis=1)
        return context

# ==========================================
# 4. BUILD ATTENTION-BASED MODEL
# ==========================================
def build_attention_model(seq_len, n_features):
    """
    Sequence-to-vector model with LSTM + Attention.
    """
    inputs = Input(shape=(seq_len, n_features))
    lstm_out = LSTM(64, return_sequences=True)(inputs)

    context_vec = Attention()(lstm_out)

    dense_out = Dense(n_features)(context_vec)

    model = Model(inputs, dense_out)
    model.compile(optimizer="adam", loss="mse")
    return model

# ==========================================
# 5. BUILD BASELINE MODEL (VANILLA LSTM)
# ==========================================
def build_baseline_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))
    lstm_out = LSTM(64)(inputs)
    outputs = Dense(n_features)(lstm_out)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# ==========================================
# 6. EVALUATION METRICS
# ==========================================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ==========================================
# 7. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    df = generate_dataset()
    data_scaled, scaler = scale_data(df)

    seq_len = 30
    X, y = create_sequences(data_scaled, seq_len)

    # Train/Val Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    n_features = df.shape[1]

    # ---- Train Attention Model ----
    att_model = build_attention_model(seq_len, n_features)
    att_model.summary()
    att_model.fit(X_train, y_train, epochs=20, batch_size=32,
                  validation_data=(X_val, y_val))

    # ---- Train Baseline LSTM ----
    base_model = build_baseline_model(seq_len, n_features)
    base_model.fit(X_train, y_train, epochs=20, batch_size=32,
                   validation_data=(X_val, y_val))

    # ---- Predictions ----
    att_pred = att_model.predict(X_val)
    base_pred = base_model.predict(X_val)

    # ---- Inverse Transform ----
    att_pred_inv = scaler.inverse_transform(att_pred)
    base_pred_inv = scaler.inverse_transform(base_pred)
    y_val_inv = scaler.inverse_transform(y_val)

    # ---- Metrics ----
    print("\nATTENTION MODEL PERFORMANCE")
    print("MAE:", mean_absolute_error(y_val_inv, att_pred_inv))
    print("RMSE:", np.sqrt(mean_squared_error(y_val_inv, att_pred_inv)))
    print("MAPE:", mape(y_val_inv, att_pred_inv))

    print("\nBASELINE LSTM PERFORMANCE")
    print("MAE:", mean_absolute_error(y_val_inv, base_pred_inv))
    print("RMSE:", np.sqrt(mean_squared_error(y_val_inv, base_pred_inv)))
    print("MAPE:", mape(y_val_inv, base_pred_inv))

    # ---- Visualization ----
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_inv[:200, 0], label="Actual")
    plt.plot(att_pred_inv[:200, 0], label="Attention Forecast")
    plt.plot(base_pred_inv[:200, 0], label="Baseline LSTM")
    plt.legend()
    plt.title("Forecast Comparison")
    plt.show()
