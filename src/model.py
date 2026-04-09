import logging
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any] | None = None) -> xgb.XGBClassifier:
    """
    Trains a cost-sensitive XGBoost classifier.
    If params are provided, they are used; otherwise, uses smart defaults.
    """
    # Default scale_pos_weight: ratio of negative samples to positive samples
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    
    if params is None:
        logger.info("Training XGBoost with default parameters and scale_pos_weight: %.2f", scale_weight)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric="aucpr",
        )
    else:
        logger.info("Training XGBoost with optimized parameters.")
        # Ensure scale_pos_weight is included if not in params
        if "scale_pos_weight" not in params:
            params["scale_pos_weight"] = scale_weight
        model = xgb.XGBClassifier(**params)
    
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the model using business-critical metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_prob)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    logger.info("Evaluation Results:")
    logger.info("Precision-Recall AUC: %.4f", pr_auc)
    logger.info("Confusion Matrix:\n%s", conf_mat)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    
    # Run Business Analysis
    business_impact_analysis(y_test.values, y_pred)

def business_impact_analysis(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """
    Calculates the financial impact of the model's performance.
    """
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Financial assumptions
    SAVINGS_PER_FRAUD = 500.0
    COST_PER_FALSE_ALARM = 20.0
    
    fraud_savings = tp * SAVINGS_PER_FRAUD
    admin_costs = fp * COST_PER_FALSE_ALARM
    net_benefit = fraud_savings - admin_costs
    
    potential_total_loss = (tp + fn) * SAVINGS_PER_FRAUD
    efficiency = (fraud_savings / potential_total_loss * 100) if potential_total_loss > 0 else 0
    
    logger.info("Business Impact Analysis:")
    logger.info("  Potential Loss Prevented: $%0.2f", fraud_savings)
    logger.info("  False Alarm Costs: $%0.2f", admin_costs)
    logger.info("  Net Financial Benefit: $%0.2f", net_benefit)
    logger.info("  Detection Efficiency: %0.2f%%", efficiency)
    
    return {
        "net_benefit": net_benefit,
        "efficiency": efficiency
    }
