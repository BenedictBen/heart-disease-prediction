"""
Master Training, Benchmarking, and Explainability Pipeline
Heart Disease Research Assessment - Graduate Portfolio Grade
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_prepare_data, FEATURE_NAMES, TARGET_NAME
from src.models import get_all_models
from src.evaluate import (
    evaluate_supervised_models,
    evaluate_unsupervised_models,
    generate_explainability_artifacts
)

def main():
    print("=" * 70)
    print(" HEART DISEASE ML RESEARCH PIPELINE (21 ALGORITHMIC PARADIGMS)")
    print("=" * 70)
    
    # 1. Load authentic dataset
    print("\n[Step 1/5] Loading Authentic UCI Cleveland Dataset...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} patient records across {len(FEATURE_NAMES)} clinical attributes.")
    
    # 2. Features and Target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    # 3. Train-Test Split (Stratified to maintain 54%/46% class distribution)
    print("\n[Step 2/5] Performing Stratified Train-Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    
    # 4. Feature Scaling (StandardScaler fit on train only)
    print("\n[Step 3/5] Fitting Feature Preprocessor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # Save Scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved feature scaler to models/scaler.pkl")
    
    # Also save feature names and test data for evaluation/tests
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump({
            "feature_names": FEATURE_NAMES,
            "target_name": TARGET_NAME,
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test
        }, f)
        
    # 5. Initialize all 21 models
    print("\n[Step 4/5] Initializing & Benchmarking 21 Algorithmic Paradigms...")
    supervised_models, unsupervised_models = get_all_models()
    
    # 5a. Supervised Evaluation & Benchmark
    benchmark_df, fitted_models = evaluate_supervised_models(
        supervised_models, X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_NAMES
    )
    
    # 5b. Unsupervised Clustering Evaluation
    unsup_df = evaluate_unsupervised_models(
        unsupervised_models, X_all_scaled, y
    )
    
    # 6. Model Explainability with SHAP (using best ensemble model: Random Forest or XGBoost)
    print("\n[Step 5/5] Generating SHAP Explainability Artifacts...")
    best_model_name = "Random_Forest" if "Random_Forest" in fitted_models else "XGBoost"
    best_model = fitted_models[best_model_name]
    generate_explainability_artifacts(
        best_model, X_train_scaled, X_test_scaled, FEATURE_NAMES, X_train
    )
    
    # Summary
    print("\n" + "=" * 70)
    print(" TRAINING & BENCHMARKING COMPLETE")
    print("=" * 70)
    print("\nTop 5 Supervised Models by ROC-AUC:")
    print(benchmark_df[["Algorithm", "Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]].head(5).to_string(index=False))
    print("\nUnsupervised Clustering Summary:")
    print(unsup_df.to_string(index=False))
    print("\nAll model binaries saved to: models/")
    print("All evaluation charts saved to: results/")
    print("=" * 70)

if __name__ == "__main__":
    main()
