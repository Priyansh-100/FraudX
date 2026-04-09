import logging
import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Trains a cost-sensitive XGBoost classifier.
    The 'scale_pos_weight' parameter is used to handle extreme imbalance.
    """
    # Calculate scale_pos_weight: ratio of negative samples to positive samples
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    
    logger.info("Training XGBoost with scale_pos_weight: %.2f", scale_weight)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="aucpr",  # Area Under PR Curve is better for fraud
    )
    
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
