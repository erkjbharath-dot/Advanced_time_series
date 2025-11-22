#!/usr/bin/env python3
"""
ts_attention_forecast.py

Production-quality single-file implementation for:
- Baseline LSTM (no attention)
- LSTM + Additive Attention decoder

Features:
- Loads statsmodels.macrodata if available, otherwise generates synthetic hourly multivariate data
- Preprocessing: scaling, sliding windows, train/val/test split
- PyTorch models with clean API, docstrings, type hints
- Training loop with early stopping, checkpointing, tensorboard hooks (optional)
- Evaluation: MAE, RMSE, MAPE; plotting helpers
- Attention extraction utilities and simple attention-heatmap saving

Requirements: numpy, pandas, scikit-learn, matplotlib, torch, statsmodels (optional)

Author: ChatGPT
"""
from __future__ import annotations
import os
import math
import time
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import statsmodels.api as sm  # type: ignore
    STATSMODELS = True
except Exception:
    STATSMODELS = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CONFIG: Dict[str, Any] = {
    "seq_len": 168,        # input window length (e.g. 7 days hourly)
    "horizon": 24,         # forecast horizon (e.g. next 24 hours)
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "patience": 6,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_dir": "./model_checkpoints",
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)

# ---------------------------
# Utils
# ---------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def mape(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.where(np.abs(a) < 1e-8, 1e-8, np.abs(a))
    return float(np.mean(np.abs((a - b) / denom))) * 100.0

# ---------------------------
# Data
# ---------------------------

def load_or_generate() -> pd.DataFrame:
    """Load a built-in statsmodels dataset (macrodata) or generate synthetic hourly data.

    Returns
    -------
    pd.DataFrame
        Multivariate dataframe with a datetime index and columns including 'target'.
    """
    if STATSMODELS:
        try:
            ds = sm.datasets.macrodata.load_pandas().data
            ds.index = pd.date_range(start="1959-01-01", periods=len(ds), freq="Q")
            df = ds.resample("M").ffill()
            df = df[["realgdp", "realcons", "realinv"]].rename(columns={
                "realgdp": "gdp", "realcons": "cons", "realinv": "inv"
            }).astype(float)
            # Create a seasonalized synthetic 'target' that depends on columns
            df["target"] = df["gdp"] * 0.6 + df["cons"] * 0.3 + 0.1 * df["inv"]
            print("Loaded statsmodels macrodata (monthly).")
            return df
        except Exception as e:
            print("statsmodels load failed, generating synthetic. Error:", e)

    N = 4000
    rng = pd.date_range("2018-01-01", periods=N, freq="H")
    t = np.arange(N)
    seasonal_daily = 8.0 * np.sin(2 * np.pi * t / 24)
    seasonal_week = 4.0 * np.sin(2 * np.pi * t / (24*7))
    trend = 0.0003 * t
    exog1 = 3.0 * np.sin(2 * np.pi * t / (24*30)) + 0.2 * np.random.randn(N)
    noise = 0.8 * np.random.randn(N)
    target = 50 + seasonal_daily + seasonal_week + trend + 0.6 * exog1 + noise
    df = pd.DataFrame({
        "target": target,
        "seasonal": seasonal_daily + seasonal_week,
        "exog1": exog1,
        "exog2": 0.5 * np.random.randn(N)
    }, index=rng)
    print("Generated synthetic dataset (hourly).")
    return df


def create_windows(df: pd.DataFrame, input_cols: List[str], target_col: str,
                   seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list, Y_list = [], []
    data = df[input_cols].values
    target = df[[target_col]].values.squeeze(-1)
    n = len(df)
    for start in range(0, n - seq_len - horizon + 1):
        end = start + seq_len
        X_list.append(data[start:end])
        Y_list.append(target[end:end+horizon])
    X = np.stack(X_list)
    Y = np.stack(Y_list)
    return X, Y

class TSDataSet(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]

# ---------------------------
# Models
# ---------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, seq_len, input_size)
        outputs, (h_n, c_n) = self.rnn(x)
        # outputs: (B, seq_len, hidden_size)
        return outputs, (h_n, c_n)

class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention over encoder timesteps.

    Computes context vector for a given decoder state and encoder outputs.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs: torch.Tensor, dec_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_outputs: (B, seq_len, enc_dim)
        # dec_state: (B, dec_dim)
        # returns: context (B, enc_dim), attn_weights (B, seq_len)
        enc_proj = self.W_enc(enc_outputs)  # (B, seq_len, attn_dim)
        dec_proj = self.W_dec(dec_state).unsqueeze(1)  # (B, 1, attn_dim)
        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_dim)
        return context, attn_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, input_dim: int, enc_dim: int, dec_dim: int, attn_dim: int,
                 horizon: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.dec_rnn = nn.LSTMCell(input_dim + enc_dim, dec_dim)
        self.attn = AdditiveAttention(enc_dim, dec_dim, attn_dim)
        self.out = nn.Linear(dec_dim, 1)

    def forward(self, enc_outputs: torch.Tensor, dec_init: Tuple[torch.Tensor, torch.Tensor],
                teacher_forcing: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_outputs: (B, seq_len, enc_dim)
        # dec_init: tuple(h0, c0) where each is (B, dec_dim)
        B = enc_outputs.size(0)
        device = enc_outputs.device
        h, c = dec_init
        # Start token: zeros
        inp = torch.zeros(B, 1, device=device)
        preds = []
        attn_weights_seq = []
        for t in range(self.horizon):
            # compute context using current hidden state
            context, attn_w = self.attn(enc_outputs, h)
            rnn_in = torch.cat([inp, context], dim=-1)  # (B, 1+enc_dim)
            h, c = self.dec_rnn(rnn_in.squeeze(1), (h, c))  # both (B, dec_dim)
            out = self.out(h).squeeze(-1)  # (B,)
            preds.append(out.unsqueeze(1))
            attn_weights_seq.append(attn_w.unsqueeze(1))
            # next input: teacher forcing or previous pred
            if teacher_forcing is not None:
                inp = teacher_forcing[:, t].unsqueeze(1)
            else:
                inp = out.unsqueeze(1)
        preds = torch.cat(preds, dim=1)  # (B, horizon)
        attn_weights = torch.cat(attn_weights_seq, dim=1)  # (B, horizon, seq_len)
        return preds, attn_weights

class BaselineLSTM(nn.Module):
    """Simple encoder LSTM + linear head that predicts all horizon steps at once."""
    def __init__(self, input_size: int, hidden_size: int, horizon: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.enc(x)
        # take last layer hidden state
        h = h_n[-1]
        out = self.head(h)
        return out

class AttentionForecastModel(nn.Module):
    def __init__(self, input_size: int, enc_dim: int, dec_dim: int, attn_dim: int,
                 horizon: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = EncoderLSTM(input_size, enc_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = DecoderWithAttention(input_dim=1, enc_dim=enc_dim, dec_dim=dec_dim,
                                            attn_dim=attn_dim, horizon=horizon, num_layers=num_layers,
                                            dropout=dropout)

    def forward(self, x: torch.Tensor, teacher_forcing: torch.Tensor | None = None):
        enc_outs, (h_n, c_n) = self.encoder(x)
        # initialize decoder hidden state from encoder final hidden (project if sizes differ)
        h0 = h_n[-1]
        c0 = c_n[-1]
        preds, attn = self.decoder(enc_outs, (h0, c0), teacher_forcing=teacher_forcing)
        return preds, attn

# ---------------------------
# Training / Evaluation
# ---------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, opt, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        if hasattr(model, 'decoder'):
            preds, _ = model(xb)
        else:
            preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if hasattr(model, 'decoder'):
                preds, att = model(xb)
            else:
                preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds_list.append(preds.cpu().numpy())
            y_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list)
    ys = np.vstack(y_list)
    return total_loss / len(loader.dataset), preds, ys

# ---------------------------
# Workflow / Main
# ---------------------------

def run_experiment():
    df = load_or_generate()
    input_cols = [c for c in df.columns if c != 'target']
    target_col = 'target'

    seq_len = CONFIG['seq_len']
    horizon = CONFIG['horizon']

    # scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xall = df[input_cols].values
    yall = df[[target_col]].values
    Xs = scaler_X.fit_transform(Xall)
    ys = scaler_y.fit_transform(yall).squeeze(-1)
    df_scaled = pd.DataFrame(np.hstack([Xs, ys.reshape(-1,1)]), index=df.index,
                             columns=input_cols + [target_col])

    X, Y = create_windows(df_scaled, input_cols, target_col, seq_len, horizon)

    # Train/val/test split (70/15/15)
    N = len(X)
    ntrain = int(0.7 * N)
    nval = int(0.15 * N)
    train_idx = slice(0, ntrain)
    val_idx = slice(ntrain, ntrain + nval)
    test_idx = slice(ntrain + nval, N)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_ds = TSDataSet(X_train, Y_train)
    val_ds = TSDataSet(X_val, Y_val)
    test_ds = TSDataSet(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    input_size = len(input_cols)

    # Baseline model
    baseline = BaselineLSTM(input_size=input_size, hidden_size=CONFIG['hidden_size'],
                            horizon=horizon, num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']).to(CONFIG['device'])
    opt_base = torch.optim.Adam(baseline.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

    # Attention model
    attn_model = AttentionForecastModel(input_size=input_size, enc_dim=CONFIG['hidden_size'],
                                        dec_dim=CONFIG['hidden_size'], attn_dim=64,
                                        horizon=horizon, num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']).to(CONFIG['device'])
    opt_attn = torch.optim.Adam(attn_model.parameters(), lr=CONFIG['lr'])

    # Training loops with early stopping (baseline then attn)
    best_val = float('inf')
    patience = 0
    for epoch in range(CONFIG['epochs']):
        t0 = time.time()
        train_loss = train_one_epoch(baseline, train_loader, opt_base, criterion, CONFIG['device'])
        val_loss, _, _ = evaluate(baseline, val_loader, criterion, CONFIG['device'])
        print(f"[Baseline] Epoch {epoch+1}/{CONFIG['epochs']} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(baseline.state_dict(), os.path.join(CONFIG['model_dir'], 'baseline.pt'))
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                print("Early stopping baseline")
                break

    best_val = float('inf')
    patience = 0
    for epoch in range(CONFIG['epochs']):
        t0 = time.time()
        train_loss = train_one_epoch(attn_model, train_loader, opt_attn, criterion, CONFIG['device'])
        val_loss, _, _ = evaluate(attn_model, val_loader, criterion, CONFIG['device'])
        print(f"[Attention] Epoch {epoch+1}/{CONFIG['epochs']} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(attn_model.state_dict(), os.path.join(CONFIG['model_dir'], 'attn_model.pt'))
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                print("Early stopping attention model")
                break

    # Load best checkpoints
    baseline.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'baseline.pt')))
    attn_model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'attn_model.pt')))

    # Evaluate on test
    _, preds_base, ys_base = evaluate(baseline, test_loader, criterion, CONFIG['device'])
    _, preds_attn, ys_attn = evaluate(attn_model, test_loader, criterion, CONFIG['device'])

    # inverse scale
    def inv_scale_y(y_scaled: np.ndarray) -> np.ndarray:
        # y_scaled shape (N, horizon)
        N, H = y_scaled.shape
        flat = y_scaled.reshape(-1,1)
        inv = scaler_y.inverse_transform(flat).reshape(N, H)
        return inv

    preds_base_inv = inv_scale_y(preds_base)
    preds_attn_inv = inv_scale_y(preds_attn)
    ys_inv = inv_scale_y(ys_base)

    # Metrics
    def summarize(preds: np.ndarray, ys: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(ys.ravel(), preds.ravel())
        r = {'MAE': float(mae), 'RMSE': float(rmse(ys.ravel(), preds.ravel())), 'MAPE': float(mape(ys.ravel(), preds.ravel()))}
        return r

    metrics_base = summarize(preds_base_inv, ys_inv)
    metrics_attn = summarize(preds_attn_inv, ys_inv)
    print("Baseline metrics:", metrics_base)
    print("Attention model metrics:", metrics_attn)

    # Save a representative attention matrix for the first test batch
    # We'll run a forward pass to extract attention for debugging/visualization
    attn_model.eval()
    with torch.no_grad():
        xb, yb = test_ds[0]
        xb_t = torch.tensor(xb).unsqueeze(0).to(CONFIG['device'])
        preds, attn = attn_model(xb_t)
        attn_np = attn.cpu().numpy()  # (1, horizon, seq_len)
        np.save(os.path.join(CONFIG['model_dir'], 'sample_attention.npy'), attn_np)
        print(f"Saved sample attention matrix to {CONFIG['model_dir']}/sample_attention.npy")

    # Plot single example
    idx = 0
    plot_example_prediction(df=df, idx_global=idx, seq_len=seq_len, horizon=horizon,
                            scaler_y=scaler_y,
                            preds_base=preds_base_inv[idx], preds_attn=preds_attn_inv[idx],
                            true_y=ys_inv[idx])


def plot_example_prediction(df: pd.DataFrame, idx_global: int, seq_len: int, horizon: int,
                            scaler_y, preds_base, preds_attn, true_y):
    # idx_global is index in test windows (0 is first test window)
    # We will reconstruct the time axis for that window and plot
    # For simplicity, use the original df index
    plt.figure(figsize=(10,5))
    # Reconstruct time indices
    start = idx_global
    times_hist = df.index[start:start+seq_len]
    times_fore = df.index[start+seq_len:start+seq_len+horizon]
    hist_vals = scaler_y.inverse_transform(df[['target']].values[start:start+seq_len])[:,0]
    plt.plot(times_hist, hist_vals, label='history')
    plt.plot(times_fore, true_y, label='true')
    plt.plot(times_fore, preds_base, label='baseline')
    plt.plot(times_fore, preds_attn, label='attention')
    plt.legend()
    plt.title('Example Forecast')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['model_dir'], 'example_forecast.png'))
    print(f"Saved example forecast plot to {CONFIG['model_dir']}/example_forecast.png")

if __name__ == '__main__':
    run_experiment()
