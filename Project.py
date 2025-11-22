"""
Advanced GBM Project
File: Advanced_GBM_Asymmetric_Loss_Project.py
Contents:
 - Production-quality pipeline for an asymmetric loss optimized XGBoost regressor
 - Uses California Housing dataset (scikit-learn)
 - Defines an asymmetric squared error loss that penalizes underestimation more strongly
 - Implements XGBoost custom objective (gradient & hessian) and custom evaluation metric
 - Trains a baseline model (standard squared error) and performs randomized hyperparameter
   optimization targeting the custom metric (manual randomized search using xgboost.train)
 - Produces a comparative summary and prints the final optimized hyperparameters

Notes:
 - This script is written defensively: if xgboost is not available it will raise an informative error
 - The hyperparameter search is simple and reproducible; swap to Hyperopt / skopt if available
 - For production, consider persisting best models with joblib or xgboost.Booster.save_model

Author: ChatGPT (for student project)
"""

import os
import math
import time
import random
import json
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import xgboost; if not available, raise a clear error so user can pip-install it
try:
    import xgboost as xgb
except Exception as e:
    raise ImportError(
        "This script requires xgboost. Install it with: pip install xgboost\nOriginal error: {}".format(e)
    )

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------
# Business scenario & loss
# ---------------------------
# Scenario: inventory/demand forecasting where underestimation (stockouts) is more costly than
# overestimation (holding cost). We'll penalize underestimates K times more than overestimates.
# Asymmetric squared error:
#   L(r) = w_pos * r^2  if r >= 0  (overestimation: pred >= true)
#        = w_neg * r^2  if r <  0  (underestimation: pred < true)
# where r = y_pred - y_true. We choose w_neg > w_pos, e.g. w_pos=1.0, w_neg=3.0

W_OVER = 1.0   # weight for overestimation
W_UNDER = 3.0  # weight for underestimation (penalize underestimates more)

# Mathematical derivatives (for XGBoost custom objective):
# g = dL/dy_pred = 2 * w * (y_pred - y_true)
# h = d^2L/dy_pred^2 = 2 * w
# where w depends on sign of (y_pred - y_true).

# ---------------------------
# Custom objective & metric
# ---------------------------

def asymmetric_obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """Custom XGBoost objective (gradient, hessian) for asymmetric squared error.

    Args:
        preds: raw predictions (float32)
        dtrain: DMatrix containing labels
    Returns:
        grad, hess arrays
    """
    labels = dtrain.get_label()
    residual = preds - labels

    # weight per sample depending on sign of residual
    w = np.where(residual < 0.0, W_UNDER, W_OVER).astype(np.float32)
    grad = 2.0 * w * residual
    hess = 2.0 * w * np.ones_like(residual)
    return grad, hess


def asymmetric_metric(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """Custom evaluation metric (average asymmetric squared error).

    Returns (name, value) where lower is better.
    """
    labels = dtrain.get_label()
    residual = preds - labels
    w = np.where(residual < 0.0, W_UNDER, W_OVER)
    loss = (w * (residual ** 2)).mean()
    return 'asym_mse', float(loss)

# A sklearn-compatible scorer (higher is better), we will use negative loss
from sklearn.metrics import make_scorer

def asym_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, w_over=W_OVER, w_under=W_UNDER) -> float:
    r = y_pred - y_true
    w = np.where(r < 0.0, w_under, w_over)
    return float((w * (r ** 2)).mean())

asym_scorer = make_scorer(lambda y_true, y_pred: -asym_loss_numpy(y_true, y_pred), greater_is_better=True)

# ---------------------------
# Data loading & preprocessing
# ---------------------------

def load_data(test_size=0.2, val_size=0.125, random_state=RNG_SEED):
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    # Split temp into val and test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - relative_val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------
# Baseline training (standard objective)
# ---------------------------

def train_baseline(X_train, y_train, X_val, y_val, num_boost_round=500, early_stopping_rounds=30):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': RNG_SEED,
        'verbosity': 0,
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster

# ---------------------------
# Train using custom objective
# ---------------------------

def train_with_custom_obj(X_train, y_train, X_val, y_val, params, num_boost_round=1000, early_stopping_rounds=30):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'val')]

    booster = xgb.train(
        params,
        dtrain,
        obj=asymmetric_obj,
        feval=asymmetric_metric,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster

# ---------------------------
# Manual randomized hyperparameter search targeting asymmetric metric
# ---------------------------

def sample_params() -> Dict[str, Any]:
    # reasonable search space for XGBoost (regression)
    return {
        'eta': float(10 ** random.uniform(-3, -0.3)),   # learning rate 0.001 - ~0.5
        'max_depth': random.randint(3, 10),
        'subsample': float(random.uniform(0.5, 1.0)),
        'colsample_bytree': float(random.uniform(0.5, 1.0)),
        'lambda': float(10 ** random.uniform(-3, 1)),  # L2 reg
        'alpha': float(10 ** random.uniform(-3, 1)),   # L1 reg
        'min_child_weight': float(10 ** random.uniform(-1, 2)),
        'seed': RNG_SEED,
        'verbosity': 0,
        'tree_method': 'auto',
    }


def randomized_search(
    X_train, y_train, X_val, y_val, n_iter=30, num_boost_round=1000, early_stopping_rounds=30
) -> Tuple[Dict[str, Any], xgb.Booster, pd.DataFrame]:
    best_score = float('inf')
    best_params = None
    best_model = None
    records = []

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    for i in range(n_iter):
        params = sample_params()
        try:
            booster = xgb.train(
                params,
                dtrain,
                obj=asymmetric_obj,
                feval=asymmetric_metric,
                num_boost_round=num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        except Exception as e:
            print(f"Iteration {i}: training failed with params {params}. Error: {e}")
            continue

        # Predict on validation set and compute asymmetric metric
        preds_val = booster.predict(dval)
        val_loss = asym_loss_numpy(y_val.values if hasattr(y_val, 'values') else y_val, preds_val)

        rec = params.copy()
        rec.update({'iter': i, 'val_asym_loss': val_loss, 'best_iteration': booster.best_iteration})
        records.append(rec)

        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_model = booster
            print(f"New best iter={i}: val_asym_loss={val_loss:.6f}")

    records_df = pd.DataFrame(records)
    return best_params, best_model, records_df

# ---------------------------
# Evaluate models & print summary
# ---------------------------

def evaluate_model(booster: xgb.Booster, X_test, y_test) -> Dict[str, float]:
    dtest = xgb.DMatrix(X_test)
    preds = booster.predict(dtest)

    metrics = {}
    metrics['asym_mse'] = asym_loss_numpy(y_test.values if hasattr(y_test, 'values') else y_test, preds)
    metrics['mae'] = mean_absolute_error(y_test, preds)
    metrics['rmse'] = math.sqrt(mean_squared_error(y_test, preds))
    metrics['r2'] = r2_score(y_test, preds)
    return metrics

# ---------------------------
# Main runnable flow
# ---------------------------
if __name__ == '__main__':
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Baseline
    print("Training baseline model (standard squared error)...")
    baseline_booster = train_baseline(X_train, y_train, X_val, y_val, num_boost_round=500)

    baseline_metrics = evaluate_model(baseline_booster, X_test, y_test)
    print('\nBaseline performance on test set:')
    print(json.dumps(baseline_metrics, indent=2))

    # Hyperparameter optimization (manual randomized search)
    print('\nStarting randomized hyperparameter search targeting asymmetric metric...')
    start_time = time.time()
    best_params, best_model, records_df = randomized_search(
        X_train, y_train, X_val, y_val, n_iter=25, num_boost_round=1000, early_stopping_rounds=50
    )
    elapsed = time.time() - start_time
    print(f"Randomized search done in {elapsed/60.0:.2f} minutes")

    if best_model is None:
        raise RuntimeError("Hyperparameter search failed to produce a model. Check logs and xgboost availability.")

    optimized_metrics = evaluate_model(best_model, X_test, y_test)
    print('\nOptimized model performance on test set:')
    print(json.dumps(optimized_metrics, indent=2))

    # Comparative summary
    summary = {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'best_params': best_params,
        'search_records_head': records_df.sort_values('val_asym_loss').head().to_dict(orient='records')
    }

    print('\nComparative summary (printed as JSON):')
    print(json.dumps(summary, indent=2, default=str))

    # Save best model and records for later inspection
    out_dir = 'gbm_asymmetry_results'
    os.makedirs(out_dir, exist_ok=True)
    best_model.save_model(os.path.join(out_dir, 'best_asymmetric_xgb.model'))
    records_df.to_csv(os.path.join(out_dir, 'search_records.csv'), index=False)

    # Final hyperparameters block (structured)
    print('\nFinal optimized hyperparameters:')
    print(json.dumps(best_params, indent=2))


# ---------------------------
# End of script
# ---------------------------

# -----------------------------------------------------------------
# Report (mathematical derivation, business interpretation, and notes)
# Placed here for convenience — extract into separate report file if desired
# -----------------------------------------------------------------

REPORT = r"""
Project: Advanced Optimization of Gradient Boosting Machines using Custom Loss Functions

1) Business interpretation
--------------------------
In inventory/demand-forecasting contexts, underestimating demand (predicting less than actual) can cause stockouts
and lost sales, which typically have a higher cost than holding excess inventory. We capture this asymmetry by
penalizing underestimates more heavily than overestimates.

2) Custom loss (asymmetric squared error)
-----------------------------------------
Let y be the true target and y_hat the prediction. Define residual r = y_hat - y.
The asymmetric squared error loss is:
    L(r) = w_pos * r^2  if r >= 0  (overestimation)
         = w_neg * r^2  if r <  0  (underestimation)
Choose weights w_neg > w_pos; e.g. w_pos=1.0, w_neg=3.0.

3) Gradient and Hessian (for GBM / XGBoost)
-------------------------------------------
We need the first and second derivatives of L with respect to y_hat.
    g = dL/dy_hat = 2 * w * (y_hat - y) = 2 * w * r
    h = d^2L/dy_hat^2 = 2 * w
where w = w_neg if r < 0 else w_pos. These closed-form expressions allow direct integration
as a custom objective in XGBoost.

4) Optimization strategy
------------------------
We integrate asymmetric_obj (returning grad/hess) and asymmetric_metric (for evaluation) into xgboost.train.
We search hyperparameters with a randomized search that samples sensible ranges for eta, max_depth,
subsample, colsample_bytree, regularization (lambda, alpha), and min_child_weight. The objective used for
training is the asymmetric objective, and the validation metric used to select best models is the asymmetric
metric (asym_mse). This ensures the search prioritizes business-relevant performance.

5) Baseline vs optimized comparison
-----------------------------------
We report:
 - asym_mse (primary metric, lower is better)
 - MAE, RMSE, R^2 (secondary metrics to show trade-offs)

Typically, the optimized model should reduce asym_mse compared to the baseline. There may be trade-offs such as
slightly worse RMSE or R^2 depending on how aggressively the model focuses on reducing costly underestimates.

6) Production notes
-------------------
 - Persist the final Booster with booster.save_model()
 - For reproducible hyperparameter search at scale, move to hyperopt/skopt/Bayesian optimization
 - Consider calibration of W_UNDER and W_OVER by mapping monetary cost ratios to the weight ratio
 - If deploying within an ML platform, wrap the custom objective and metric in the training scaffold used
   by your infra (for example, use callbacks or a training wrapper that injects the objective function)

"""

# Save report as a file for convenience
with open(os.path.join('gbm_asymmetry_results' if os.path.exists('gbm_asymmetry_results') else '.', 'report.txt'), 'w') as f:
    f.write(REPORT)

print('\nA plaintext report was written to ./report.txt (also included in the script variable REPORT).')
