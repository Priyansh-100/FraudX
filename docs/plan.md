# Project Implementation Plan: FraudX

## Objective
Develop a robust financial fraud detection system that effectively identifies rare fraudulent transactions (less than 1%) while minimizing false positives, utilizing modern Python development standards.

## Phase 1: Environment and Project Structure (COMPLETED)
1. Initialize the project using `uv` for dependency management and environment isolation.
2. Set up a flat `src/` layout for simple, modular access to pipeline components.
3. Configure `Ruff` and `Mypy (strict)` for high-quality code and type safety.
4. Establish IDE-alignment via `pyrefly.toml` and `pyrightconfig.json`.

## Phase 2: Data Ingestion and Exploration (COMPLETED)
1. Implement a high-fidelity synthetic data generator in `src/data_generator.py`.
2. Generate an imbalanced dataset (0.5% fraud ratio) with 30 features (PCA-like V-features + Time + Amount).
3. Implement basic ingestion and logging in the main orchestrator.

## Phase 3: Preprocessing and Feature Engineering (COMPLETED)
1. Apply Log-transformation to the `Amount` feature to reduce skewness.
2. Transform `Time` into cyclic Sine/Cosine components (hour-based) to capture daily patterns.
3. Implement `RobustScaler` to handle outliers in PCA features.
4. Ensure stratified train-test splitting to maintain fraud ratios.

## Phase 4: Model Development and Tuning (COMPLETED)
1. Implement a cost-sensitive `XGBoost` classifier using `scale_pos_weight`.
2. Integrate `Optuna` for automated hyperparameter optimization (n=20 trials).
3. Target **PR-AUC** as the primary optimization metric for imbalanced data.

## Phase 5: Evaluation and Business Metrics (COMPLETED)
1. Evaluate using PR-AUC, Confusion Matrix, and F1-Score.
2. Implement **Business Impact Analysis** to calculate real-world financial benefit:
   - Potential Loss Prevented ($500 per TP)
   - False Alarm Costs ($20 per FP)
   - Net Financial Benefit estimation.

## Phase 6: Full Pipeline Integration (COMPLETED)
1. Create a unified entry point in `src/main.py`.
2. Enable optional hyperparameter tuning within the main execution flow.
3. Output comprehensive logs for each pipeline stage.

## Phase 7: Quality Assurance and Maintenance (STRETCH)
1. Maintain zero-error status for `Ruff` and `Mypy`.
2. (Next) Implement unit tests in `tests/` using `pytest`.
3. (Next) Add model persistence (saving `.json` or `.joblib` models).
4. (Next) Dockerize the pipeline for consistent execution.
