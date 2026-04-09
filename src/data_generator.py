import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_synthetic_data(output_path: Path, n_samples: int = 10000, fraud_ratio: float = 0.005) -> None:
    """
    Generates a realistic synthetic credit card fraud dataset.
    Mimics typical fraud data characteristics: high dimensionality, 
    extreme class imbalance (0.5% fraud), and distinct distributions.
    """
    logger.info("Generating %d samples with %.2f%% fraud ratio...", n_samples, fraud_ratio * 100)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # 30 V-features (anonymized principal components)
    n_features = 28
    
    # Legit transactions: mean 0, unit variance
    legit_features = np.random.normal(0, 1, size=(n_legit, n_features))
    
    # Fraud transactions: shifted distributions for some features
    fraud_features = np.random.normal(0.5, 1.5, size=(n_fraud, n_features))
    # Some features are explicitly more discriminatory
    fraud_features[:, [3, 10, 14, 17]] += 2.0 
    
    X = np.vstack([legit_features, fraud_features])
    
    # Time and Amount
    time = np.random.uniform(0, 172800, size=n_samples)
    amount = np.random.exponential(scale=100, size=n_samples)
    
    X = np.hstack([time.reshape(-1, 1), X, amount.reshape(-1, 1)])
    
    # Labels
    y = np.array([0] * n_legit + [1] * n_fraud)
    
    # Create DataFrame
    cols = ["Time"] + [f"V{i+1}" for i in range(n_features)] + ["Amount"]
    df = pd.DataFrame(X, columns=cols)
    df["Class"] = y
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved to %s", output_path)

if __name__ == "__main__":
    # Testing generation
    logging.basicConfig(level=logging.INFO)
    generate_synthetic_data(Path("data/creditcard.csv"))
