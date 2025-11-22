"""
time_series_attention_project.py

Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms.

- Generates a synthetic multivariate time series (>=5000 points).
- Implements baseline LSTM and attention-based Transformer encoder model.
- Performs walk-forward validation and compares models (MAE, RMSE, MAPE).
- Extracts and visualizes attention weights for interpretability.
- Includes basic hyperparameter grid sweep for sequence length and attention heads.

Author: ChatGPT (for educational project)
Date: 2025-11-22

Requirements
------------
torch, numpy, pandas, matplotlib, scikit-learn, tqdm

Example
-------
python time_series_attention_project.py
"""

import os
import math
import random
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Utilities & Metrics
# -----------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (MAPE). Avoid zero division by small epsilon."""
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -----------------------------
# Synthetic Data Generation
# -----------------------------
def generate_synthetic_multivariate_ts(n_points: int = 10000,
                                       n_vars: int = 4,
                                       seed: int = 42) -> pd.DataFrame:
    """
    Generate a complex multivariate time series combining multiple frequencies,
    non-stationarity, regime changes, and noise.

    Parameters
    ----------
    n_points : int
        Number of time steps (>=5000 recommended).
    n_vars : int
        Number of variables (features).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['t', 'var0', 'var1', ..., 'target'] where 'target' is the
        primary forecast target (a nonlinear combination of vars).
    """
    np.random.seed(seed)
    t = np.arange(n_points)
    df = pd.DataFrame({'t': t})

    # Base signals: multiple sinusoids with varying frequencies + trend + jumps
    freqs = np.linspace(0.01, 0.5, num=n_vars)
    for i in range(n_vars):
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        seasonal = amplitude * np.sin(2 * np.pi * freqs[i] * t + phase)
        multi_freq = 0.5 * np.sin(2 * np.pi * (freqs[i] * 3) * t + phase / 2)
        trend = 0.0001 * (t ** 1.5) * (1 + 0.1 * np.sin(0.001 * t + i))
        noise = 0.5 * np.random.randn(n_points)
        regime = (np.tanh(0.0005 * (t - n_points // 2)) + 1) * 0.5  # smooth regime shift
        series = (seasonal + multi_freq) * regime + trend + noise
        df[f'var{i}'] = series

    # Create a nonlinear target that depends on past variables
    # e.g., target = var0 * var1 shifted + nonlinear transform + noise
    df['target'] = (0.6 * df['var0'] + 0.3 * df['var1'] ** 2 -
                    0.2 * np.roll(df['var2'], 3) + 0.1 * np.sin(df['var3'])) \
                   + 0.3 * np.random.randn(n_points)

    # Optionally drop initial rows with rolled values introduced by np.roll
    df = df.iloc[10:].reset_index(drop=True)
    return df

# -----------------------------
# PyTorch Dataset for Sequence Data
# -----------------------------
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset to serve sliding-window sequences for multivariate forecasting.

    Attributes
    ----------
    sequences : np.ndarray
        Array of input sequences shaped (N, seq_len, n_features).
    targets : np.ndarray
        Array of target values shaped (N, target_dim).
    """

    def __init__(self, data: pd.DataFrame, feature_cols: List[str], target_col: str,
                 seq_len: int = 64, pred_horizon: int = 1):
        """
        Initialize dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Time-indexed data.
        feature_cols : List[str]
            Names of feature columns used as input.
        target_col : str
            Name of target column to forecast.
        seq_len : int
            Length of the input sequence.
        pred_horizon : int
            Forecast horizon (1 means next step).
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if pred_horizon <= 0:
            raise ValueError("pred_horizon must be > 0")
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        X = data[feature_cols].values.astype(np.float32)
        y = data[target_col].values.astype(np.float32)

        self.sequences = []
        self.targets = []

        max_start = len(data) - seq_len - pred_horizon + 1
        if max_start <= 0:
            raise ValueError("Data too short for the chosen seq_len and pred_horizon.")

        for start in range(max_start):
            seq = X[start:start + seq_len]
            target = y[start + seq_len + pred_horizon - 1]
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.stack(self.sequences)  # shape (N, seq_len, n_features)
        self.targets = np.array(self.targets).reshape(-1, 1)  # shape (N, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq = torch.from_numpy(self.sequences[idx])
        tgt = torch.from_numpy(self.targets[idx])
        return seq, tgt

# -----------------------------
# Models
# -----------------------------
class LSTMForecast(nn.Module):
    """
    Standard LSTM-based regressor.

    Args
    ----
    input_dim: int
        Number of input features.
    hidden_dim: int
        Hidden dimension of LSTM.
    num_layers: int
        Number of LSTM layers.
    dropout: float
        Dropout between LSTM layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        last = out[:, -1, :]  # take last timestep hidden state
        return self.fc(last)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (same as original Transformer).
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerForecast(nn.Module):
    """
    Transformer encoder-based forecaster that returns attention weights from the last encoder block.

    Args
    ----
    input_dim: int
        Number of input features.
    d_model: int
        Model dimension (will linearly project input_dim -> d_model).
    nhead: int
        Number of attention heads.
    num_layers: int
        Number of Transformer encoder layers.
    dim_feedforward: int
        Feedforward dimension inside Transformer.
    dropout: float
        Dropout probability.
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        # We will keep layers in a list to be able to extract attn weights
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        # Placeholder where we'll store attention weights during forward pass
        self._last_
