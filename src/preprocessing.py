import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares the fraud dataset for modeling.
    Handles scaling and train-test split with stratification.
    """
    logger.info("Preprocessing data: shape %s", df.shape)
    
    # Class balance check
    class_counts = df["Class"].value_counts(normalize=True)
    logger.info("Class distribution: \n%s", class_counts)
    
    # Standardize Time and Amount (as they have different scales than V-features)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])
    
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Stratified split is crucial for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info("Split complete. Train size: %d, Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
