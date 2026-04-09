# FraudX: Intelligent Fraud Detection Pipeline

FraudX is a high-performance financial fraud detection system designed to identify rare fraudulent transactions within extremely imbalanced datasets. Built with modern Python standards and a focus on financial impact, it bridges the gap between machine learning metrics and real-world business value.

## Key Features

*   **Advanced Feature Engineering**: Includes Log-transformation for skewed amounts and Cyclic (Sine/Cosine) encoding for temporal patterns.
*   **Automated Hyperparameter Optimization**: Integrated Optuna engine that searches for the optimal XGBoost configuration to maximize Precision-Recall AUC.
*   **Business Impact Analysis**: Evaluation includes financial metrics such as Potential Loss Prevented and False Alarm Costs.
*   **Modern Workspace**: Managed by uv with a flat src/ layout for high developer velocity.
*   **Rigorous QA**: Zero-tolerance policy for linting and type errors via Ruff and Mypy (Strict).


## Getting Started

### Prerequisites
*   Python 3.13+
*   uv installed on your system.

### Installation
```bash
# Clone the repository
git clone https://github.com/Priyansh-100/FraudX.git
cd FraudX

# Synchronize dependencies and environment
uv sync --all-extras
```

### Running the Pipeline
```bash
# Execute the full end-to-end pipeline (Data -> Tuning -> Eval)
uv run src/main.py
```

## Current Performance Baseline

The model is evaluated on a synthetic dataset with a 0.5% fraud ratio.

| Metric | Value |
| :--- | :--- |
| **Precision-Recall AUC** | ~0.82 |
| **Recall (Fraud Detection)** | 60% - 70% |
| **False Positives** | < 0.1% |
| **Net Financial Benefit** | ~$2,960 / 2k tx |

## Quality Assurance

To maintain high code quality standards, run the following checks:

```bash
# Linting & Formatting
uv run ruff check src

# Strict Type Checking
uv run mypy src
```