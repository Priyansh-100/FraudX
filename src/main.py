import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from data_generator import generate_synthetic_data
from model import evaluate_model, train_model
from preprocessing import preprocess_data
from tuner import tune_hyperparameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("fraudx")

def run_pipeline(tune: bool = True) -> None:
    """Orchestrates the FraudX end-to-end detection pipeline."""
    data_path = Path("data/creditcard.csv")
    
    # 1. Data Generation
    if not data_path.exists():
        logger.info("Dataset not found. Generating synthetic data...")
        generate_synthetic_data(data_path)
    
    # 2. Ingestion
    df = pd.read_csv(data_path)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 4. Hyperparameter Tuning (Optional)
    best_params: dict[str, Any] | None = None
    if tune:
        best_params = tune_hyperparameters(X_train, y_train)
    
    # 5. Training with optimized parameters
    model = train_model(X_train, y_train, params=best_params)
    
    # 6. Final Evaluation
    evaluate_model(model, X_test, y_test)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    # Run with tuning enabled by default.
    run_pipeline(tune=True)
