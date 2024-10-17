# Project: Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an end-to-end machine learning pipeline using **DVC (Data Version Control)** for data and model versioning, and **MLflow** for experiment tracking. The pipeline focuses on training a **Random Forest Classifier** on the **Pima Diabetes Dataset**, with clear stages for data preprocessing, model training, and evaluation.

## Data Version Control (DVC)

- **DVC** is used to track and version the dataset, models, and pipeline stages, ensuring reproducibility across different environments.
- The pipeline is structured into stages (preprocessing, training, evaluation) that can be automatically re-executed if any dependencies change (e.g., data, scripts, or parameters).
- **DVC** also allows remote data storage (e.g., DagsHub, S3) for large datasets and models.

## Experiment Tracking with MLflow

- **MLflow** is used to track experiment metrics, parameters, and artifacts.
- It logs the hyperparameters of the model (e.g., `n_estimators`, `max_depth`) and performance metrics like accuracy.
- **MLflow** helps compare different runs and models to optimize the machine learning pipeline.

---

## Pipeline Stages

### 1. Preprocessing

- The `preprocess.py` script reads the raw dataset (`data/raw/data.csv`), performs basic preprocessing (such as renaming columns), and outputs the processed data to `data/processed/data.csv`.
- This stage ensures that data is consistently processed across runs.

### 2. Training

- The `train.py` script trains a **Random Forest Classifier** on the preprocessed data.
- The model is saved as `models/random_forest.pkl`.
- Hyperparameters and the model itself are logged into **MLflow** for tracking and comparison.

### 3. Evaluation

- The `evaluate.py` script loads the trained model and evaluates its performance (accuracy) on the dataset.
- The evaluation metrics are logged to **MLflow** for tracking.

---

## Technology Stack

- **Python**: The core programming language for data processing, model training, and evaluation.
- **DVC**: For version control of data, models, and pipeline stages.
- **MLflow**: For logging and tracking experiments, metrics, and model artifacts.
- **Scikit-learn**: For building and training the **Random Forest Classifier**.
