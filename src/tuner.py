import logging
from typing import Any

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Optuna objective function for tuning XGBoost parameters.
    Targets PR-AUC as it is the most robust metric for fraud detection.
    """
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": scale_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_val)[:, 1]
    score = float(average_precision_score(y_val, y_prob))
    
    return score

def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """
    Orchestrates the Optuna study to find the best XGBoost hyperparameters.
    """
    logger.info("Starting hyperparameter tuning with Optuna...")
    
    # Internal split for validation during tuning
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_t, y_t, X_v, y_v), n_trials=20)
    
    logger.info("Tuning complete. Best Score: %.4f", study.best_value)
    logger.info("Best Parameters: %s", study.best_params)
    
    return study.best_params
