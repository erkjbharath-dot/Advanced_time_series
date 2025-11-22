"""
Advanced Multivariate Time Series Forecasting with Attention

This script:
1. Generates a synthetic multivariate time series dataset
2. Preprocesses data (scaling, missing values, windowing)
3. Implements baseline LSTM and an Attention-based Seq2Seq model
4. Trains both models
5. Evaluates using MAE, RMSE, MAPE
6. Plots predictions and attention weights

Dependencies:
- numpy
- pandas
- sklearn
- tensorflow>=2.8
- matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------
# 1. Generate Dataset
# --------------------------------------------------

def generate_dataset(n_steps=1500):
    """Generate a 5-feature synthetic multivariate time series."""
    t = np.arange(n_steps)

    temp = 10 + 5*np.sin(2*np.pi*t/50) + 0.02*t + np.random.normal(0,0.3,n_steps)
    pressure = 1000 + 2*np.sin(2*np.pi*t/75) + np.random.normal(0,0.2,n_steps)
    humidity = 50 + 0.4*temp + np.random.normal(0,1,n_steps)
    wind_speed = 3 + 0.005*t + np.random.normal(0,0.1,n_steps)
    power_demand = 0.3*temp + 0.2*humidity + 0.1*pressure + np.random.normal(0,2,n_steps)

    df = pd.DataFrame({
        "temp": temp,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "power_demand": power_demand
    })
    return df


df = generate_dataset()
print("Dataset head:")
print(df.head())

# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------

def fill_missing(df):
    """Production-ready missing value filler."""
    return df.ffill().bfill()


def scale_features(df):
    """Scale features using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, history_size, target_size):
    """Convert time series into input/output sequences for forecasting."""
    X, y = [], []
    for i in range(len(data) - history_size - target_size):
        X.append(data[i:i+history_size])
        y.append(data[i+history_size:i+history_size+target_size, -1])  # predict power_demand
    return np.array(X), np.array(y)


df = fill_missing(df)
data_scaled, scaler = scale_features(df)

HISTORY = 48
TARGET = 12
X, y = create_sequences(data_scaled, HISTORY, TARGET)

# Time split
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]


# --------------------------------------------------
# 3. Baseline LSTM Model
# --------------------------------------------------

def build_baseline_lstm():
    inputs = layers.Input(shape=(HISTORY, X.shape[2]))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    outputs = layers.Dense(TARGET)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer=Adam(1e-3))
    return model


baseline = build_baseline_lstm()
baseline.summary()

history_baseline = baseline.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)


# --------------------------------------------------
# 4. Attention-based model
# --------------------------------------------------

class BahdanauAttention(layers.Layer):
    """Standard Bahdanau attention."""

    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query: (batch, hidden)
        # values: (batch, time, hidden)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


def build_attention_model():
    encoder_inputs = layers.Input(shape=(HISTORY, X.shape[2]))
    encoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    decoder_inputs = layers.Input(shape=(TARGET, X.shape[2]))
    decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
    attention = BahdanauAttention(32)

    all_outputs = []
    decoder_state_h, decoder_state_c = state_h, state_c

    for t in range(TARGET):
        context, att_weights = attention(decoder_state_h, encoder_outputs)

        decoder_input_t = decoder_inputs[:, t:t+1, :]
        decoder_input_t = tf.concat([decoder_input_t, tf.expand_dims(context,1)], axis=-1)

        output, decoder_state_h, decoder_state_c = decoder_lstm(
            decoder_input_t, initial_state=[decoder_state_h, decoder_state_c]
        )

        out = layers.Dense(1)(output)
        all_outputs.append(out)

    outputs = layers.Concatenate(axis=1)(all_outputs)
    outputs = layers.Reshape((TARGET,))(outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


attention_model = build_attention_model()
attention_model.summary()

# Decoder inputs are zeros for training (teacher forcing alternative)
decoder_zero = np.zeros((len(X_train), TARGET, X.shape[2]))
decoder_zero_val = np.zeros((len(X_val), TARGET, X.shape[2]))

history_attention = attention_model.fit(
    [X_train, decoder_zero], y_train,
    validation_data=([X_val, decoder_zero_val], y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model, X, y, decoder_input=None):
    preds = model.predict(X if decoder_input is None else [X, decoder_input])
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    mape = np.mean(np.abs((y - preds) / y)) * 100
    return preds, mae, rmse, mape


decoder_zero_test = np.zeros((len(X_test), TARGET, X.shape[2]))

pred_lstm, mae_lstm, rmse_lstm, mape_lstm = evaluate(baseline, X_test, y_test)
pred_attn, mae_attn, rmse_attn, mape_attn = evaluate(
    attention_model, X_test, y_test, decoder_zero_test
)

print("\n--- BASELINE LSTM ---")
print("MAE:", mae_lstm)
print("RMSE:", rmse_lstm)
print("MAPE:", mape_lstm)

print("\n--- ATTENTION MODEL ---")
print("MAE:", mae_attn)
print("RMSE:", rmse_attn)
print("MAPE:", mape_attn)


# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(y_test[:100,0], label="Actual")
plt.plot(pred_attn[:100,0], label="Attention Prediction")
plt.plot(pred_lstm[:100,0], label="LSTM Prediction")
plt.legend()
plt.title("Forecast Comparison")
plt.show()
