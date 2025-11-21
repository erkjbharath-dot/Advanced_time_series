"""
Advanced Time Series Forecasting with Deep Learning and Explainability
Single-file implementation (LSTM) with:
- Synthetic multivariate dataset generation (non-stationary, trend, seasonality, regime change, interactions)
- Windowing / scaling
- LSTM model + small hyperparameter grid search + early stopping
- Baseline linear autoregression (flattened lags, OLS)
- Integrated Gradients interpretability for three high-error examples
- Evaluation (RMSE / MAE) and plots

Run instructions:
1) Recommended environment: Python 3.9+
2) Install dependencies:
   pip install numpy scipy pandas matplotlib scikit-learn tensorflow==2.14.1
   (If using Colab, TensorFlow is preinstalled; choose a recent runtime.)
3) Save this as `time_series_lstm_explain.py` and run:
   python time_series_lstm_explain.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os

# -------------------------------
# 1) Synthetic multivariate dataset
# -------------------------------
def generate_synthetic(T=6000, seed=42):
    np.random.seed(seed)
    t = np.arange(T)

    # Feature A: demand-like: trend + multiple seasonalities + noise + regime change
    trend = 0.0008 * t
    season_weekly = 2.0 * np.sin(2 * np.pi * t / 168)       # weekly
    season_multi = 1.5 * np.sin(2 * np.pi * t / 876)       # longer period
    noise = 0.6 * np.random.randn(T)
    regime = (t > int(0.58 * T)).astype(float) * (0.8 * np.sin(2 * np.pi * t / 40))

    feat1 = 10 + trend + season_weekly + season_multi + regime + noise

    # Feature B: exogenous (price/temperature) that interacts with feat1
    season_b = 1.2 * np.cos(2 * np.pi * t / 168 + 0.5)
    trend_b = -0.0002 * t
    interaction_noise = 0.3 * np.random.randn(T)
    feat2 = 5 + trend_b + season_b + 0.5 * (feat1 - feat1.mean())/feat1.std() + interaction_noise

    # Target: future demand with nonlinear interactions and extra short seasonality
    target = (feat1 * 0.6) + 0.4 * feat2 + 0.3 * np.roll(feat1, 1) * (1 + 0.05*np.sin(2*np.pi*t/50))
    target += 0.8 * np.sin(2*np.pi*(t+10)/30)
    target += 0.9 * (np.random.randn(T) * 0.4)

    # trim first few because of roll
    start = 10
    df = pd.DataFrame({
        'feat_demand': feat1[start:],
        'feat_exog': feat2[start:],
        'target': target[start:]
    })
    df.index = np.arange(len(df))
    return df

# -------------------------------
# 2) Windowing & scaling helpers
# -------------------------------
def create_windows(arr, seq_len, horizon=1):
    # arr: np array columns: [feat1, feat2, target]
    X, y = [], []
    total = len(arr)
    for i in range(total - seq_len - (horizon-1)):
        X.append(arr[i:i+seq_len, :2])               # two features as inputs
        y.append(arr[i+seq_len + (horizon-1), 2])    # predict target at horizon
    return np.array(X), np.array(y)

def prepare_dataset_split(df, seq_len, scaler_X, scaler_y):
    vals = df[['feat_demand','feat_exog','target']].values
    X, y = create_windows(vals, seq_len)
    # scale X and y
    Xflat = X.reshape(-1, 2)
    Xflat_s = scaler_X.transform(Xflat)
    Xs = Xflat_s.reshape(X.shape)
    ys = scaler_y.transform(y.reshape(-1,1)).reshape(-1)
    return Xs, ys

# -------------------------------
# 3) LSTM model builder
# -------------------------------
def build_lstm(seq_len, hidden_units=64, lr=1e-3):
    inp = layers.Input(shape=(seq_len, 2))
    x = layers.LSTM(hidden_units, return_sequences=False)(inp)
    x = layers.Dense(max(16, hidden_units//2), activation='relu')(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# -------------------------------
# 4) Training + hyperparameter tuning (small grid)
# -------------------------------
def grid_search_lstm(train_df, val_df, candidate_seq_lens=[24,48], hidden_options=[32,64], lrs=[1e-3,5e-4]):
    results = []
    # fit scalers on train only
    scaler_X = StandardScaler().fit(train_df[['feat_demand','feat_exog']])
    scaler_y = StandardScaler().fit(train_df[['target']])
    for seq_len in candidate_seq_lens:
        X_tr, y_tr = prepare_dataset_split(train_df, seq_len, scaler_X, scaler_y)
        X_val, y_val = prepare_dataset_split(val_df, seq_len, scaler_X, scaler_y)
        for hidden in hidden_options:
            for lr in lrs:
                tf.keras.backend.clear_session()
                model = build_lstm(seq_len, hidden_units=hidden, lr=lr)
                es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
                hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=60, batch_size=64, callbacks=[es], verbose=0)
                val_pred = model.predict(X_val).reshape(-1)
                # convert back to real units to compute val RMSE
                val_pred_un = scaler_y.inverse_transform(val_pred.reshape(-1,1)).reshape(-1)
                val_y_un = scaler_y.inverse_transform(y_val.reshape(-1,1)).reshape(-1)
                val_rmse = np.sqrt(mean_squared_error(val_y_un, val_pred_un))
                val_mae = mean_absolute_error(val_y_un, val_pred_un)
                results.append({'seq_len':seq_len,'hidden':hidden,'lr':lr,'val_RMSE':val_rmse,'val_MAE':val_mae,'epochs':len(hist.history['loss']),'scaler_X':scaler_X,'scaler_y':scaler_y,'model':model})
                print(f"[grid] seq={seq_len} hidden={hidden} lr={lr} -> val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} epochs={len(hist.history['loss'])}")
    # choose best
    best = sorted(results, key=lambda x: x['val_RMSE'])[0]
    return best, results

# -------------------------------
# 5) Baseline linear autoregression (flattened lags)
# -------------------------------
def build_linear_ar(df_train, df_test, seq_len, scaler_X):
    # create flattened lag features (no scaling in scaler applied to each time-step pair)
    def create_flat_lag(df):
        vals = df[['feat_demand','feat_exog','target']].values
        Xflat, yflat = [], []
        for i in range(len(vals) - seq_len):
            Xflat.append(vals[i:i+seq_len, :2].flatten())
            yflat.append(vals[i+seq_len, 2])
        return np.array(Xflat), np.array(yflat)
    X_tr, y_tr = create_flat_lag(df_train)
    X_te, y_te = create_flat_lag(df_test)
    # scale feature pairs by scaler_X per time step
    X_tr_s = X_tr.reshape(-1,2)
    X_tr_s = scaler_X.transform(X_tr_s).reshape(X_tr.shape)
    X_te_s = X_te.reshape(-1,2)
    X_te_s = scaler_X.transform(X_te_s).reshape(X_te.shape)
    lr = LinearRegression().fit(X_tr_s, y_tr)
    pred_te = lr.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(y_te, pred_te))
    mae = mean_absolute_error(y_te, pred_te)
    return lr, rmse, mae, pred_te, y_te

# -------------------------------
# 6) Integrated Gradients for the LSTM
# -------------------------------
def integrated_gradients_keras(model, input_sequence, baseline=None, m_steps=50):
    """
    Returns integrated gradients for a single input sequence (seq_len, features).
    Model outputs a scalar (scaled target). We compute gradients of the scalar wrt input.
    Works for Keras models (TF backend).
    """
    if baseline is None:
        baseline = np.zeros_like(input_sequence).astype(np.float32)
    input_sequence = input_sequence.astype(np.float32)
    baseline = baseline.astype(np.float32)
    alphas = np.linspace(0.0, 1.0, m_steps+1)
    total_grad = np.zeros_like(input_sequence, dtype=np.float32)
    for alpha in alphas:
        interp = baseline + alpha * (input_sequence - baseline)
        interp_tf = tf.convert_to_tensor(interp.reshape((1,) + interp.shape))
        with tf.GradientTape() as tape:
            tape.watch(interp_tf)
            pred = model(interp_tf)  # returns tensor shape (1,1)
        grad = tape.gradient(pred, interp_tf).numpy()[0]  # shape (seq_len, features)
        total_grad += grad
    avg_grad = total_grad / (m_steps+1)
    ig = (input_sequence - baseline) * avg_grad
    return ig  # same shape as input_sequence

# -------------------------------
# 7) Main orchestration
# -------------------------------
def main():
    # generate data
    df = generate_synthetic(T=6000)
    Tlen = len(df)
    # splits: train / val / test (time-ordered)
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(Tlen * train_frac)
    n_val = int(Tlen * val_frac)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    print("Splits:", len(train_df), len(val_df), len(test_df))

    # grid search
    candidate_seq_lens = [24, 48]
    hidden_options = [32, 64]
    lrs = [1e-3, 5e-4]

    best, all_results = grid_search_lstm(train_df, val_df, candidate_seq_lens, hidden_options, lrs)

    print("BEST HYPERPARAMS (grid):", {k:best[k] for k in ['seq_len','hidden','lr','val_RMSE','val_MAE','epochs']})

    # retrain best model on train+val
    seq_len = best['seq_len']
    scaler_X = best['scaler_X']
    scaler_y = best['scaler_y']
    X_train_full, y_train_full = prepare_dataset_split(pd.concat([train_df, val_df]), seq_len, scaler_X, scaler_y)
    X_test, y_test = prepare_dataset_split(test_df, seq_len, scaler_X, scaler_y)

    tf.keras.backend.clear_session()
    model = build_lstm(seq_len, hidden_units=best['hidden'], lr=best['lr'])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    # use small val split for early stopping
    hist = model.fit(X_train_full, y_train_full, validation_split=0.1, epochs=80, batch_size=64, callbacks=[es], verbose=1)

    # predictions & metrics (rescale)
    pred_test = model.predict(X_test).reshape(-1)
    pred_test_un = scaler_y.inverse_transform(pred_test.reshape(-1,1)).reshape(-1)
    y_test_un = scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_test_un, pred_test_un))
    mae = mean_absolute_error(y_test_un, pred_test_un)
    print(f"LSTM Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # baseline linear AR
    lr_model, rmse_lr, mae_lr, pred_lr, y_te_lr = build_linear_ar(pd.concat([train_df, val_df]), test_df, seq_len, scaler_X)
    print(f"Linear AR Test RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}")

    # find top-3 highest residuals (challenges)
    residuals = np.abs(y_test_un - pred_test_un)
    top_idxs = residuals.argsort()[-3:][::-1]
    print("Top-3 challenging indices (in X_test):", top_idxs)

    # compute integrated gradients for these examples
    ig_results = []
    for idx in top_idxs:
        sample_input = X_test[idx]  # scaled inputs shape (seq_len, 2)
        ig = integrated_gradients_keras(model, sample_input, baseline=np.zeros_like(sample_input), m_steps=50)
        # convert input back to raw units for plots
        input_raw = scaler_X.inverse_transform(sample_input.reshape(-1,2)).reshape(sample_input.shape)
        # convert IG to approximate target-units by dividing by output scaling (rough approximation)
        ig_in_target_units = ig * (1.0 / scaler_y.scale_[0])
        ig_results.append({'idx':idx, 'ig_scaled':ig, 'ig_target':ig_in_target_units, 'input_raw':input_raw, 'pred':pred_test_un[idx], 'true':y_test_un[idx]})

    # Plot predictions vs truth (sample slice)
    n_plot = min(500, len(y_test_un))
    plt.figure(figsize=(12,4))
    plt.plot(y_test_un[:n_plot], label='True')
    plt.plot(pred_test_un[:n_plot], label='LSTM Pred')
    # plot linear AR (it may have slightly different alignment, but we plotted truncated)
    lr_plot = pred_lr[:n_plot]
    plt.plot(lr_plot, label='Linear AR Pred')
    plt.legend()
    plt.title("Test: True vs LSTM vs Linear AR (first {} points)".format(n_plot))
    plt.show()

    # Print summary table
    summary = pd.DataFrame([
        {'model':'LSTM', 'RMSE':rmse, 'MAE':mae},
        {'model':'Linear_AR', 'RMSE':rmse_lr, 'MAE':mae_lr}
    ])
    print(summary.to_string(index=False))

    # Plot integrated gradients heatmaps + input signals for the three examples
    for i, r in enumerate(ig_results):
        ig_t = r['ig_target']  # shape (seq_len, 2)
        inp = r['input_raw']   # shape (seq_len,2)
        idx = r['idx']
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        axs[0].plot(inp[:,0])
        axs[0].set_title(f'Example {i+1} (index {idx}) - input feat_demand (raw)')
        axs[1].plot(inp[:,1])
        axs[1].set_title('input feat_exog (raw)')
        im = axs[2].imshow(ig_t.T, aspect='auto')
        axs[2].set_title('Integrated Gradients approximate contributions (rows: feat_demand, feat_exog)')
        axs[2].set_xlabel('time steps (older -> newer)')
        fig.colorbar(im, ax=axs[2])
        plt.tight_layout()
        plt.show()

    # Save models/summary if desired
    out_dir = "ts_lstm_output"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "best_lstm_model"))
    summary.to_csv(os.path.join(out_dir, "performance_summary.csv"), index=False)
    print("Saved model and outputs to", out_dir)

    # Return objects for further programmatic analysis
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'best_params': best,
        'test_preds': pred_test_un,
        'test_truth': y_test_un,
        'lr_pred': pred_lr,
        'ig_results': ig_results,
        'summary': summary
    }

if __name__ == "__main__":
    results = main()
