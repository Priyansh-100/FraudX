import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs feature engineering and scaling on the fraud dataset.
    
    Includes:
    1. Log transformation of 'Amount' to handle skewness.
    2. Cyclic transformation of 'Time' (Hour component) to capture temporal patterns.
    3. Robust scaling of PCA features to handle potential outliers.
    """
    logger.info("Starting advanced feature engineering...")
    
    # 1. Amount Transformation: Log scale to normalize distribution
    df["Log_Amount"] = np.log1p(df["Amount"])
    
    # 2. Time Transformation: Extract hour and convert to cyclic features
    # Assuming 'Time' is seconds from the start of the data
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    # Drop raw Time/Amount and temporary Hour column
    df = df.drop(columns=["Time", "Amount", "Hour"])
    
    # Define features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Stratified Split to maintain fraud ratios (0.5%)
    logger.info("Performing stratified train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Robust Scaling: Better for data with outliers than StandardScaler
    # We fit only on training data to prevent leakage
    scaler = RobustScaler()
    
    # Get columns to scale (V1-V28 and Log_Amount)
    cols_to_scale = [col for col in X_train.columns if col.startswith("V") or col == "Log_Amount"]
    
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    logger.info("Preprocessing complete. Final features: %s", list(X_train.columns))
    
    return X_train, X_test, y_train, y_test
