# ============================
# ADVANCED TIME SERIES PROJECT
# One-Page Complete Python Code
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------
# 1) DATA GENERATION (5 features, 1500 observations)
# ----------------------------------------------------
np.random.seed(42)
T = 1500
t = np.arange(T)

data = pd.DataFrame({
    "load": 50 + 10*np.sin(0.02*t) + np.random.normal(0,1,T),
    "temp": 25 + 8*np.sin(0.01*t+1) + np.random.normal(0,1,T),
    "humidity": 60 + 5*np.cos(0.015*t) + np.random.normal(0,1,T),
    "wind": 10 + 3*np.sin(0.03*t+2),
    "pressure": 1000 + 20*np.cos(0.005*t)
})

# Normalize
df = (data - data.mean()) / data.std()

# ----------------------------------------------------
# Helper: Create sequences
# ----------------------------------------------------
def create_sequences(df, seq_len=30):
    X, y = [], []
    arr = df.values
    for i in range(len(arr)-seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, 0])  # predict "load"
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(df, SEQ_LEN)

# Train/val split
split = int(len(X)*0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ----------------------------------------------------
# 2) BASELINE LSTM MODEL
# ----------------------------------------------------
def baseline_lstm():
    inp = Input((SEQ_LEN, 5))
    x = LSTM(64)(inp)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

baseline = baseline_lstm()
baseline.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

# ----------------------------------------------------
# 3) ATTENTION LSTM MODEL
# ----------------------------------------------------
class Attention(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight("W", shape=(input_shape[-1],1))
        self.b = self.add_weight("b", shape=(1,))
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)      # (batch, seq, 1)
        a = tf.nn.softmax(e, axis=1)                                 # attention scores
        out = tf.reduce_sum(a * x, axis=1)                           # weighted sum
        return out, a

def attention_model():
    inp = Input((SEQ_LEN, 5))
    x = LSTM(64, return_sequences=True)(inp)
    context, attn = Attention()(x)
    out = Dense(1)(context)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

att_model = attention_model()
att_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

# Extract attention model for inference
att_layer = [l for l in att_model.layers if isinstance(l, Attention)][0]

# ----------------------------------------------------
# 4) EVALUATION METRICS
# ----------------------------------------------------
def evaluate(model, X, y):
    pred = model.predict(X, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    mape = np.mean(np.abs((y - pred) / y)) * 100
    return rmse, mae, mape

bl_rmse, bl_mae, bl_mape = evaluate(baseline, X_val, y_val)
at_rmse, at_mae, at_mape = evaluate(att_model, X_val, y_val)

print("\n=== BASELINE LSTM ===")
print("RMSE:", bl_rmse, " MAE:", bl_mae, " MAPE:", bl_mape)

print("\n=== ATTENTION MODEL ===")
print("RMSE:", at_rmse, " MAE:", at_mae, " MAPE:", at_mape)

# ----------------------------------------------------
# 5) ATTENTION VISUALIZATION
# ----------------------------------------------------
sample = X_val[0:1]
_, attn_weights = att_layer(sample)

plt.figure(figsize=(8,3))
plt.plot(attn_weights.numpy().reshape(-1))
plt.title("Attention Weights Over Time (Sample Input)")
plt.xlabel("Time Step")
plt.ylabel("Attention")
plt.show()
