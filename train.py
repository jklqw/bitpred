"""Machine Learning Training and Validation Pipeline for GDELT News + Bitcoin Price Prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from features import build_ml_dataset


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Separate feature matrix X and target vectors y (directional target & continuous return).
    Identifies all engineered news and market features.
    """
    # Columns to exclude from feature matrix
    target_cols = [
        "target_next_return", "target_direction", "target_next_volatility",
        "open", "high", "low", "close"
    ]
    feature_cols = [col for col in df.columns if col not in target_cols]

    X = df[feature_cols].copy()
    y_direction = df["target_direction"].copy()
    y_return = df["target_next_return"].copy()

    # Replace any infinities or NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y_direction, y_return, feature_cols


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    y_ret: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Strict chronological split without shuffling to prevent temporal data leakage.
    Order: [ Train (past) ] -> [ Validation ] -> [ Test (unseen future) ]
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train, ret_train = X.iloc[:train_end], y.iloc[:train_end], y_ret.iloc[:train_end]
    X_val, y_val, ret_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end], y_ret.iloc[train_end:val_end]
    X_test, y_test, ret_test = X.iloc[val_end:], y.iloc[val_end:], y_ret.iloc[val_end:]

    return {
        "X_train": X_train, "y_train": y_train, "ret_train": ret_train,
        "X_val": X_val, "y_val": y_val, "ret_val": ret_val,
        "X_test": X_test, "y_test": y_test, "ret_test": ret_test
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate key classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc
    }


def backtest_strategy(returns: pd.Series, signals: np.ndarray) -> Dict[str, float]:
    """
    Simulate a simple long/cash strategy based on model predictions.
    If signal == 1: Long BTC
    If signal == 0: Cash (0% return)
    """
    strat_returns = returns.values * signals
    bh_cumulative = np.prod(1 + returns.values) - 1
    strat_cumulative = np.prod(1 + strat_returns) - 1

    # Annualized Sharpe (assuming hourly intervals: 24 * 365 = 8760 periods/year)
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(8760)

    return {
        "Strategy_Cumulative_Return": float(strat_cumulative),
        "Buy_and_Hold_Return": float(bh_cumulative),
        "Strategy_Sharpe_Ratio": float(sharpe),
        "Total_Trades": int(np.sum(np.diff(signals) != 0))
    }


def train_and_evaluate_models(freq: str = "1h") -> Dict[str, Any]:
    """
    Run the end-to-end ML training workflow:
    1. Build feature dataset
    2. Split chronologically
    3. Scale features using training distribution only
    4. Train Logistic Regression, Random Forest, HistGradientBoosting
    5. Evaluate against naive & momentum baselines
    6. Extract category feature importance
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    print(f"[*] Building dataset with frequency={freq}...")
    dataset = build_ml_dataset(freq=freq)
    if len(dataset) < 30:
        raise ValueError(f"Insufficient data samples ({len(dataset)}). Need at least 30 intervals to train.")

    X, y, ret, feature_cols = prepare_features_and_targets(dataset)
    print(f"[*] Extracted {X.shape[0]} timestamps and {X.shape[1]} features.")

    split = chronological_split(X, y, ret, train_ratio=0.70, val_ratio=0.15)
    print(f"[*] Split: Train={len(split['X_train'])}, Val={len(split['X_val'])}, Test={len(split['X_test'])}")

    # Standardize features (fit on Train ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split["X_train"])
    X_val_scaled = scaler.transform(split["X_val"])
    X_test_scaled = scaler.transform(split["X_test"])

    # Baselines
    majority_class = int(split["y_train"].mode()[0])
    y_test_majority = np.full(len(split["y_test"]), majority_class)

    # Momentum baseline (predicts previous period sign)
    y_test_momentum = (split["X_test"]["btc_log_return"].values > 0).astype(int)

    # Define Candidate Models
    models = {
        "Majority_Baseline": None,
        "Momentum_Baseline": None,
        "Logistic_Regression": LogisticRegression(C=0.1, max_iter=500, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42),
        "Gradient_Boosting": HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=5, random_state=42)
    }

    results = {}
    for name, model in models.items():
        if name == "Majority_Baseline":
            pred = y_test_majority
            prob = np.full(len(pred), 0.5)
        elif name == "Momentum_Baseline":
            pred = y_test_momentum
            prob = np.full(len(pred), 0.5)
        else:
            model.fit(X_train_scaled, split["y_train"])
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = evaluate_predictions(split["y_test"].values, pred, prob)
        strat_metrics = backtest_strategy(split["ret_test"], pred)
        results[name] = {**metrics, **strat_metrics}

    # Extract Feature Importance (from Logistic Regression & Random Forest)
    lr_model = models["Logistic_Regression"]
    rf_model = models["Random_Forest"]

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "LR_Coefficient": lr_model.coef_[0],
        "RF_Importance": rf_model.feature_importances_
    }).sort_values(by="RF_Importance", ascending=False)

    return {
        "model_results": pd.DataFrame(results).T,
        "feature_importance": coef_df,
        "dataset_summary": {
            "total_samples": len(dataset),
            "feature_count": len(feature_cols),
            "date_start": str(dataset.index.min()),
            "date_end": str(dataset.index.max())
        }
    }
