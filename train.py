"""Machine Learning Training, Walk-Forward Validation, Ensemble Modeling, and Realistic Backtesting."""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from config import (
    DATA_DIR,
    RANDOM_SEED,
    DEFAULT_INTERVAL,
    DEFAULT_PREDICTION_HORIZON,
    DEFAULT_TARGET_THRESHOLD,
    DEFAULT_TAKER_FEE,
    DEFAULT_SLIPPAGE
)
from features import build_ml_dataset, diagnose_feature_coverage

MODEL_ARTIFACT_PATH = DATA_DIR / "best_prediction_model.joblib"


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Separate feature matrix X, binary direction target y, and future continuous trade return.
    Excludes non-feature columns, raw price levels, and targets.
    """
    excluded_cols = [
        "target_future_return", "target_direction", "target_close_to_close",
        "open", "high", "low", "close", "quote_volume"
    ]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()
    y_direction = df["target_direction"].copy()
    y_return = df["target_future_return"].copy()

    # Clean any accidental NaNs or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y_direction, y_return, feature_cols


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    y_ret: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Strict 3-way chronological split without shuffling:
    [ Train (70%) ] -> [ Validation (15%) ] -> [ Untouched Test (15%) ]
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train": X.iloc[:train_end], "y_train": y.iloc[:train_end], "ret_train": y_ret.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end], "y_val": y.iloc[train_end:val_end], "ret_val": y_ret.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:], "y_test": y.iloc[val_end:], "ret_test": y_ret.iloc[val_end:]
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics:
    Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Confusion_Matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    }


def realistic_backtest(
    returns: pd.Series,
    signals: np.ndarray,
    taker_fee: float = DEFAULT_TAKER_FEE,
    slippage: float = DEFAULT_SLIPPAGE
) -> Dict[str, float]:
    """
    Simulate realistic trading strategy with transaction fees, slippage, and execution timing.
    - Signal S_t in {0, 1} enters at t+1 Open.
    - Transaction cost applied on position entry AND exit: fee + slippage per side.
    """
    cost_per_side = taker_fee + slippage
    signals = np.asarray(signals, dtype=float)
    ret_arr = returns.values

    # Position turnover
    position_changes = np.abs(np.diff(signals, prepend=0.0))
    transaction_costs = position_changes * cost_per_side

    # Net strategy period returns
    strategy_returns = (signals * ret_arr) - transaction_costs

    # Buy & Hold return after costs
    bh_gross_cumulative = np.prod(1.0 + ret_arr) - 1.0
    bh_net_cumulative = (1.0 - cost_per_side) * (1.0 + bh_gross_cumulative) * (1.0 - cost_per_side) - 1.0

    # Strategy cumulative curve
    cum_curve = np.cumprod(1.0 + strategy_returns)
    strat_net_cumulative = float(cum_curve[-1] - 1.0) if len(cum_curve) > 0 else 0.0

    # Maximum Drawdown
    peaks = np.maximum.accumulate(cum_curve)
    drawdowns = (peaks - cum_curve) / (peaks + 1e-9)
    max_mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Annualized Sharpe (hourly = sqrt(8760))
    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns))
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(8760)

    # Win Rate and Profit Factor
    active_mask = (signals > 0)
    active_returns = ret_arr[active_mask]
    if len(active_returns) > 0:
        win_rate = float(np.sum(active_returns > 0) / len(active_returns))
        total_gains = float(np.sum(active_returns[active_returns > 0]))
        total_losses = float(np.abs(np.sum(active_returns[active_returns < 0])))
        profit_factor = (total_gains / (total_losses + 1e-9)) if total_losses > 0 else 1.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    total_trades = int(np.sum(position_changes > 0) // 2)

    return {
        "Net_Return": round(strat_net_cumulative * 100.0, 2),
        "BH_Net_Return": round(bh_net_cumulative * 100.0, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown": round(max_mdd * 100.0, 2),
        "Win_Rate": round(win_rate * 100.0, 2),
        "Profit_Factor": round(profit_factor, 2),
        "Total_Trades": total_trades
    }


def walk_forward_validation(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    ret_dev: pd.Series,
    n_splits: int = 3
) -> pd.DataFrame:
    """
    Perform expanding-window walk-forward validation across the development period (Train + Val).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    n = len(X_dev)
    split_size = n // (n_splits + 1)
    fold_results = []

    for fold in range(1, n_splits + 1):
        train_idx = split_size * fold
        val_idx = min(n, split_size * (fold + 1))

        X_tr = X_dev.iloc[:train_idx]
        y_tr = y_dev.iloc[:train_idx]
        X_v = X_dev.iloc[train_idx:val_idx]
        y_v = y_dev.iloc[train_idx:val_idx]
        ret_v = ret_dev.iloc[train_idx:val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)

        maj_pred = np.full(len(y_v), int(y_tr.mode()[0]))
        maj_acc = float(np.mean(maj_pred == y_v.values))

        # Model: Gradient Boosting
        gb = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, min_samples_leaf=15, random_state=RANDOM_SEED)
        gb.fit(X_tr_s, y_tr)
        gb_pred = gb.predict(X_v_s)
        gb_acc = float(np.mean(gb_pred == y_v.values))
        gb_bt = realistic_backtest(ret_v, gb_pred)

        fold_results.append({
            "Fold": fold,
            "Train_Samples": len(X_tr),
            "Val_Samples": len(X_v),
            "Majority_Acc": round(maj_acc, 4),
            "GB_Acc": round(gb_acc, 4),
            "GB_Sharpe": gb_bt["Sharpe_Ratio"],
            "GB_Net_Return": gb_bt["Net_Return"]
        })

    return pd.DataFrame(fold_results)


class SoftVotingEnsemble:
    """Soft Voting Ensemble combining Gradient Boosting, Random Forest, and Logistic Regression."""
    def __init__(self, models: List[Tuple[str, Any, float]]):
        self.models = models  # [(name, model, weight)]

    def predict_proba(self, X):
        prob = np.zeros(len(X))
        total_w = sum(w for _, _, w in self.models)
        for _, m, w in self.models:
            prob += (w / total_w) * m.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


def train_and_evaluate_pipeline(
    freq: str = DEFAULT_INTERVAL,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    overlap_only: bool = False,
    run_walk_forward: bool = True,
    taker_fee: float = DEFAULT_TAKER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end robust modeling workflow with Ensemble & Model Serialization.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    np.random.seed(RANDOM_SEED)

    print(f"[*] Building dataset (horizon={prediction_horizon}h, threshold={target_threshold}, overlap_only={overlap_only})...")
    dataset, metadata = build_ml_dataset(
        freq=freq,
        prediction_horizon=prediction_horizon,
        target_threshold=target_threshold,
        overlap_only=overlap_only,
        db_path=db_path
    )

    # 1. Data Sufficiency Checks
    if len(dataset) < 100:
        raise ValueError(
            f"Insufficient data samples: {len(dataset)}. Minimum required is 100. "
            "Ingest more BTC candles using --fetch-btc [N]."
        )
    elif len(dataset) < 300:
        warnings.warn(f"Dataset is small ({len(dataset)} samples). Test metrics may have high variance.")

    # 2. Extract Features & Targets
    X, y, ret, feature_cols = prepare_features_and_targets(dataset)
    coverage_df = diagnose_feature_coverage(dataset)

    # 3. Chronological Split (70% Train, 15% Val, 15% Untouched Test)
    split = chronological_split(X, y, ret, train_ratio=0.70, val_ratio=0.15)

    n_train = len(split["X_train"])
    n_val = len(split["X_val"])
    n_test = len(split["X_test"])

    tr_bal = float(split["y_train"].mean())
    val_bal = float(split["y_val"].mean())
    test_bal = float(split["y_test"].mean())

    balance_info = {
        "Train_Size": n_train, "Train_Positive_Pct": round(tr_bal * 100, 2),
        "Val_Size": n_val, "Val_Positive_Pct": round(val_bal * 100, 2),
        "Test_Size": n_test, "Test_Positive_Pct": round(test_bal * 100, 2),
    }

    if tr_bal < 0.20 or tr_bal > 0.80:
        warnings.warn(f"Severe class imbalance in training set: {tr_bal * 100:.1f}% positive.")

    # 4. Preprocessing: Fit Scaler on Train ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split["X_train"])
    X_val_scaled = scaler.transform(split["X_val"])
    X_test_scaled = scaler.transform(split["X_test"])

    # 5. Candidate Models & Tuning on Validation
    candidate_configs = {
        "Logistic_Regression": [
            LogisticRegression(C=0.01, max_iter=500, random_state=RANDOM_SEED),
            LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED),
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        ],
        "Random_Forest": [
            RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=15, random_state=RANDOM_SEED),
            RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=RANDOM_SEED),
            RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=RANDOM_SEED)
        ],
        "Gradient_Boosting": [
            HistGradientBoostingClassifier(learning_rate=0.02, max_depth=3, min_samples_leaf=15, random_state=RANDOM_SEED),
            HistGradientBoostingClassifier(learning_rate=0.04, max_depth=3, min_samples_leaf=10, random_state=RANDOM_SEED),
            HistGradientBoostingClassifier(learning_rate=0.08, max_depth=4, min_samples_leaf=10, random_state=RANDOM_SEED)
        ]
    }

    best_models = {}
    val_model_reports = {}

    # Baselines on Validation
    majority_class = int(split["y_train"].mode()[0])
    y_val_majority = np.full(n_val, majority_class)
    val_model_reports["Majority_Baseline"] = {
        **compute_metrics(split["y_val"].values, y_val_majority, np.full(n_val, 0.5)),
        **realistic_backtest(split["ret_val"], y_val_majority, taker_fee, slippage)
    }

    momentum_col = "btc_return_1h" if "btc_return_1h" in split["X_val"].columns else feature_cols[0]
    y_val_momentum = (split["X_val"][momentum_col].values > 0).astype(int)
    val_model_reports["Momentum_Baseline"] = {
        **compute_metrics(split["y_val"].values, y_val_momentum, np.full(n_val, 0.5)),
        **realistic_backtest(split["ret_val"], y_val_momentum, taker_fee, slippage)
    }

    for model_name, configs in candidate_configs.items():
        best_cfg = None
        best_val_score = -999.0
        best_thresh = 0.5
        best_val_metrics = None

        for cfg in configs:
            cfg.fit(X_train_scaled, split["y_train"])
            val_probs = cfg.predict_proba(X_val_scaled)[:, 1]

            for thresh in [0.45, 0.50, 0.55]:
                val_preds = (val_probs >= thresh).astype(int)
                val_m = compute_metrics(split["y_val"].values, val_preds, val_probs)
                val_bt = realistic_backtest(split["ret_val"], val_preds, taker_fee, slippage)

                val_score = val_m["Balanced_Accuracy"] + 0.1 * np.clip(val_bt["Sharpe_Ratio"], -2.0, 5.0)

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_cfg = cfg
                    best_thresh = thresh
                    best_val_metrics = {**val_m, **val_bt, "Optimal_Threshold": thresh}

        best_models[model_name] = (best_cfg, best_thresh)
        val_model_reports[model_name] = best_val_metrics

    # 6. Build Soft Voting Ensemble
    gb_model, gb_th = best_models["Gradient_Boosting"]
    rf_model, rf_th = best_models["Random_Forest"]
    lr_model, lr_th = best_models["Logistic_Regression"]

    ensemble = SoftVotingEnsemble([
        ("Gradient_Boosting", gb_model, 0.50),
        ("Random_Forest", rf_model, 0.35),
        ("Logistic_Regression", lr_model, 0.15)
    ])

    val_ensemble_probs = ensemble.predict_proba(X_val_scaled)[:, 1]
    best_ens_thresh = 0.50
    best_ens_score = -999.0
    best_ens_val_metrics = None

    for thresh in [0.46, 0.48, 0.50, 0.52, 0.54]:
        ens_preds = (val_ensemble_probs >= thresh).astype(int)
        v_m = compute_metrics(split["y_val"].values, ens_preds, val_ensemble_probs)
        v_bt = realistic_backtest(split["ret_val"], ens_preds, taker_fee, slippage)
        v_sc = v_m["Balanced_Accuracy"] + 0.1 * np.clip(v_bt["Sharpe_Ratio"], -2.0, 5.0)
        if v_sc > best_ens_score:
            best_ens_score = v_sc
            best_ens_thresh = thresh
            best_ens_val_metrics = {**v_m, **v_bt, "Optimal_Threshold": thresh}

    val_model_reports["Voting_Ensemble"] = best_ens_val_metrics
    best_models["Voting_Ensemble"] = (ensemble, best_ens_thresh)

    # 7. Final Evaluation on Untouched Test Set
    test_model_reports = {}

    y_test_majority = np.full(n_test, majority_class)
    test_model_reports["Majority_Baseline"] = {
        **compute_metrics(split["y_test"].values, y_test_majority, np.full(n_test, 0.5)),
        **realistic_backtest(split["ret_test"], y_test_majority, taker_fee, slippage)
    }

    y_test_momentum = (split["X_test"][momentum_col].values > 0).astype(int)
    test_model_reports["Momentum_Baseline"] = {
        **compute_metrics(split["y_test"].values, y_test_momentum, np.full(n_test, 0.5)),
        **realistic_backtest(split["ret_test"], y_test_momentum, taker_fee, slippage)
    }

    for model_name, (trained_model, chosen_thresh) in best_models.items():
        test_probs = trained_model.predict_proba(X_test_scaled)[:, 1]
        test_preds = (test_probs >= chosen_thresh).astype(int)

        t_metrics = compute_metrics(split["y_test"].values, test_preds, test_probs)
        t_backtest = realistic_backtest(split["ret_test"], test_preds, taker_fee, slippage)

        test_model_reports[model_name] = {
            **t_metrics,
            **t_backtest,
            "Used_Threshold": chosen_thresh
        }

    # Selective / High-Confidence Trading Metrics on Test set
    test_ens_probs = ensemble.predict_proba(X_test_scaled)[:, 1]
    high_conf_mask = (test_ens_probs >= 0.53) | (test_ens_probs <= 0.47)
    if np.sum(high_conf_mask) > 10:
        conf_preds = (test_ens_probs[high_conf_mask] >= 0.50).astype(int)
        conf_acc = float(np.mean(conf_preds == split["y_test"].values[high_conf_mask]))
        conf_bt = realistic_backtest(split["ret_test"].iloc[high_conf_mask], conf_preds, taker_fee, slippage)
        test_model_reports["Ensemble_High_Confidence"] = {
            "Accuracy": round(conf_acc, 4),
            "Balanced_Accuracy": round(conf_acc, 4),
            "Precision": round(float(np.mean(split['y_test'].values[high_conf_mask][conf_preds == 1])), 4) if np.sum(conf_preds == 1) > 0 else 0.0,
            "Recall": 0.0,
            "F1_Score": 0.0,
            "ROC_AUC": round(float(compute_metrics(split['y_test'].values[high_conf_mask], conf_preds, test_ens_probs[high_conf_mask])['ROC_AUC']), 4),
            **conf_bt,
            "Used_Threshold": "P>=0.53 or P<=0.47",
            "Confusion_Matrix": f"Active_Samples={np.sum(high_conf_mask)}/{n_test} ({np.sum(high_conf_mask)/n_test*100:.1f}%)"
        }

    # 8. Feature Importance Extraction
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Type": ["News" if ((not c.startswith("btc_")) and any(k in c for k in ["news", "tone", "vol_", "share_", "neg_", "pos_", "has_news"])) else "Market" for c in feature_cols],
        "RF_Importance": rf_model.feature_importances_,
        "LR_Coefficient": lr_model.coef_[0]
    }).sort_values(by="RF_Importance", ascending=False)

    # 9. Walk-Forward Cross Validation
    wf_df = None
    if run_walk_forward:
        X_dev = pd.concat([split["X_train"], split["X_val"]])
        y_dev = pd.concat([split["y_train"], split["y_val"]])
        ret_dev = pd.concat([split["ret_train"], split["ret_val"]])
        wf_df = walk_forward_validation(X_dev, y_dev, ret_dev, n_splits=3)

    # 10. Save Best Trained Ensemble Artifact for Live Predictions
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X)
    gb_model.fit(X_full_scaled, y)
    rf_model.fit(X_full_scaled, y)
    lr_model.fit(X_full_scaled, y)

    full_ensemble = SoftVotingEnsemble([
        ("Gradient_Boosting", gb_model, 0.50),
        ("Random_Forest", rf_model, 0.35),
        ("Logistic_Regression", lr_model, 0.15)
    ])

    artifact = {
        "scaler": full_scaler,
        "ensemble": full_ensemble,
        "feature_cols": feature_cols,
        "threshold": best_ens_thresh,
        "prediction_horizon": prediction_horizon,
        "date_trained": str(dataset.index.max())
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    print(f"[+] Serialized production ensemble model artifact to {MODEL_ARTIFACT_PATH}")

    return {
        "metadata": metadata,
        "balance_info": balance_info,
        "coverage_report": coverage_df,
        "validation_results": pd.DataFrame(val_model_reports).T,
        "test_results": pd.DataFrame(test_model_reports).T,
        "feature_importance": feature_importance_df,
        "walk_forward_results": wf_df
    }


def get_x_period_artifact_path(x_days: float):
    """Path to the persisted X-day period model artifact."""
    tag = f"{x_days:g}d"
    return DATA_DIR / f"best_x_period_model_{tag}.joblib"


def train_x_period_pipeline(
    x_days: float = 4.0,
    step_hours: int = 4,
    taker_fee: float = DEFAULT_TAKER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train and evaluate machine learning models specifically for X-day period prediction:
    - Inputs: First X/2 days of current period and X/2 days of previous week.
    - Target: Price(end of X days) - Price(start of X day).
    - Preserves chronological integrity (70% train, 15% val, 15% untouched test).
    - Tunes classification thresholds on validation data.
    - Evaluates out-of-sample directional accuracy, ROC-AUC, and fee-adjusted trading backtest.
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingRegressor
    )
    from sklearn.linear_model import LogisticRegression
    from features import build_x_period_dataset

    np.random.seed(RANDOM_SEED)

    print(f"\n[*] Building X-Day Period dataset (X = {x_days} days, step = {step_hours}h)...")
    dataset, metadata = build_x_period_dataset(x_days=x_days, step_hours=step_hours, db_path=db_path)

    if len(dataset) < 100:
        raise ValueError(f"Insufficient period samples ({len(dataset)}). Please fetch more BTC candles.")

    # Exclude target and raw price tracking columns from feature matrix
    excluded_cols = ['p_start', 'p_curr', 'p_end', 'delta_p', 'delta_p_pct', 'target_dir']
    feature_cols = [c for c in dataset.columns if c not in excluded_cols]

    X = dataset[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = dataset['target_dir'].copy()
    ret = dataset['delta_p_pct'].copy()

    # Chronological Split: 70% Train, 15% Validation, 15% Untouched Test
    n = len(dataset)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    ret_train = ret.iloc[:n_train]

    X_val = X.iloc[n_train:n_val]
    y_val = y.iloc[n_train:n_val]
    ret_val = ret.iloc[n_train:n_val]

    X_test = X.iloc[n_val:]
    y_test = y.iloc[n_val:]
    ret_test = ret.iloc[n_val:]

    print(f"[*] Partitions: Train={len(X_train)} ({y_train.mean()*100:.1f}% Up) | Val={len(X_val)} ({y_val.mean()*100:.1f}% Up) | Test={len(X_test)} ({y_test.mean()*100:.1f}% Up)")

    # Robust scaling fitted strictly on train fold
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 1. HistGradientBoosting Classifier
    hgb = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=2.0
    )
    hgb.fit(X_train_s, y_train)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    rf.fit(X_train_s, y_train)

    # 3. Extra Trees Classifier
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    et.fit(X_train_s, y_train)

    # 4. Regularized Logistic Regression
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    lr.fit(X_train_s, y_train)

    # 5. Continuous Delta Regressor
    reg = HistGradientBoostingRegressor(
        random_state=RANDOM_SEED,
        max_depth=5,
        min_samples_leaf=15
    )
    reg.fit(X_train_s, ret_train)

    # Soft Voting Ensemble
    models = [
        ("Hist_Gradient_Boosting", hgb, 0.35),
        ("Random_Forest", rf, 0.35),
        ("Extra_Trees", et, 0.20),
        ("Logistic_Regression", lr, 0.10)
    ]
    ensemble = SoftVotingEnsemble(models)

    # Threshold optimization on Validation Partition
    val_probs = ensemble.predict_proba(X_val_s)[:, 1]
    best_th = 0.50
    best_val_bacc = 0.0

    for th in np.linspace(0.35, 0.65, 31):
        th_preds = (val_probs >= th).astype(int)
        bacc = compute_metrics(y_val.values, th_preds, val_probs)["Balanced_Accuracy"]
        if bacc > best_val_bacc:
            best_val_bacc = bacc
            best_th = float(th)

    # Evaluate all models on Untouched Test Partition
    test_reports = {}
    for name, model, _ in models:
        test_p = model.predict_proba(X_test_s)[:, 1]
        test_pred = (test_p >= best_th).astype(int)
        m = compute_metrics(y_test.values, test_pred, test_p)
        bt = realistic_backtest(ret_test, test_pred, taker_fee=taker_fee, slippage=slippage)
        m.update(bt)
        test_reports[name] = m

    # Ensemble Test Evaluation
    test_probs = ensemble.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= best_th).astype(int)
    ens_metrics = compute_metrics(y_test.values, test_preds, test_probs)
    ens_bt = realistic_backtest(ret_test, test_preds, taker_fee=taker_fee, slippage=slippage)
    ens_metrics.update(ens_bt)
    test_reports["Ensemble_Model"] = ens_metrics

    # Baselines: Majority class and Buy & Hold
    maj_val = int(y_train.mode()[0])
    maj_preds = np.full(len(y_test), maj_val)
    maj_metrics = compute_metrics(y_test.values, maj_preds, np.full(len(y_test), float(maj_val)))
    maj_bt = realistic_backtest(ret_test, maj_preds, taker_fee=taker_fee, slippage=slippage)
    maj_metrics.update(maj_bt)
    test_reports["Majority_Baseline"] = maj_metrics

    # Dual-confirmation High Conviction Check
    test_pred_ret = reg.predict(X_test_s)
    conf_mask_up = (test_probs >= max(best_th, 0.50)) & (test_pred_ret > 0.0)
    conf_mask_down = (test_probs < min(best_th, 0.50)) & (test_pred_ret < 0.0)
    conf_samples = conf_mask_up | conf_mask_down
    if conf_samples.sum() > 0:
        y_test_conf = y_test[conf_samples]
        preds_conf = np.where(conf_mask_up[conf_samples], 1, 0)
        from sklearn.metrics import accuracy_score
        high_conv_acc = float(accuracy_score(y_test_conf, preds_conf))
        high_conv_count = int(conf_samples.sum())
    else:
        high_conv_acc = float(ens_metrics["Accuracy"])
        high_conv_count = len(y_test)

    # Feature Importance (Random Forest)
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "RF_Importance": np.round(importances, 4),
        "Type": ["News" if "news" in c or "tone" in c else "Market" for c in feature_cols]
    }).sort_values(by="RF_Importance", ascending=False)

    # Re-fit final model on full dataset for live production inference
    full_scaler = RobustScaler()
    X_full_s = full_scaler.fit_transform(X)
    hgb.fit(X_full_s, y)
    rf.fit(X_full_s, y)
    et.fit(X_full_s, y)
    lr.fit(X_full_s, y)
    reg.fit(X_full_s, ret)

    full_ensemble = SoftVotingEnsemble([
        ("Hist_Gradient_Boosting", hgb, 0.35),
        ("Random_Forest", rf, 0.35),
        ("Extra_Trees", et, 0.20),
        ("Logistic_Regression", lr, 0.10)
    ])

    artifact_path = get_x_period_artifact_path(x_days)
    artifact = {
        "x_days": x_days,
        "scaler": full_scaler,
        "ensemble": full_ensemble,
        "regressor": reg,
        "feature_cols": feature_cols,
        "optimal_threshold": best_th,
        "test_metrics": ens_metrics,
        "high_conv_acc": high_conv_acc,
        "high_conv_count": high_conv_count,
        "total_test_samples": len(y_test),
        "date_trained": str(dataset.index.max())
    }
    joblib.dump(artifact, artifact_path)
    print(f"[+] Serialized X-day model artifact ({x_days} days) to {artifact_path}")

    return {
        "metadata": metadata,
        "test_results": pd.DataFrame(test_reports).T,
        "feature_importance": fi_df,
        "optimal_threshold": best_th,
        "high_conv_acc": high_conv_acc,
        "high_conv_count": high_conv_count,
        "artifact_path": str(artifact_path)
    }

