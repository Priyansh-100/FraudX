import logging
import sys
from pathlib import Path
import pandas as pd

from data_generator import generate_synthetic_data
from preprocessing import preprocess_data
from model import train_model, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("fraudx")

def run_pipeline() -> None:
    """Orchestrates the FraudX end-to-end detection pipeline."""
    data_path = Path("data/creditcard.csv")
    
    # 1. Data Generation (Phase 1)
    if not data_path.exists():
        logger.info("Dataset not found. Generating synthetic data...")
        generate_synthetic_data(data_path)
    
    # 2. Ingestion
    df = pd.read_csv(data_path)
    
    # 3. Preprocessing (Phase 3)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 4. Training (Phase 4)
    model = train_model(X_train, y_train)
    
    # 5. Evaluation (Phase 5)
    evaluate_model(model, X_test, y_test)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
