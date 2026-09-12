"""
Master Training, Benchmarking, Hyperparameter Tuning, and Explainability Pipeline
Heart Disease Research Assessment - Graduate Portfolio Grade
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_prepare_data, FEATURE_NAMES, TARGET_NAME
from src.models import (
    get_all_models,
    tune_top_models,
    build_stacking_classifier
)
from src.evaluate import (
    evaluate_supervised_models,
    evaluate_unsupervised_models,
    generate_explainability_artifacts,
    generate_calibration_curves,
    evaluate_demographic_slices
)

def main():
    print("=" * 70)
    print(" HEART DISEASE ML RESEARCH PIPELINE (TUNED ENSEMBLES & CLINICAL CDSS)")
    print("=" * 70)
    
    # 1. Load authentic dataset
    print("\n[Step 1/6] Loading Authentic UCI Cleveland Dataset...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} patient records across {len(FEATURE_NAMES)} clinical attributes.")
    
    # 2. Features and Target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    # 3. Train-Test Split (Stratified to maintain 54%/46% class distribution)
    print("\n[Step 2/6] Performing Stratified Train-Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    
    # 4. Feature Scaling (StandardScaler fit on train only to prevent leakage)
    print("\n[Step 3/6] Fitting Feature Preprocessor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # Save Scaler and Metadata
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved feature scaler to models/scaler.pkl")
    
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump({
            "feature_names": FEATURE_NAMES,
            "target_name": TARGET_NAME,
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test
        }, f)
        
    # 5. Automated Hyperparameter Optimization
    print("\n[Step 4/6] Running Automated Hyperparameter Tuning (5-Fold CV)...")
    tuned_models, tuning_summary = tune_top_models(X_train_scaled, y_train, n_iter=8)
    
    # Save tuning summary
    tuning_df = pd.DataFrame(tuning_summary)
    os.makedirs("results", exist_ok=True)
    tuning_df.to_csv("results/hyperparameter_tuning_summary.csv", index=False)
    print("Saved hyperparameter tuning report to results/hyperparameter_tuning_summary.csv")
    
    # Initialize all models and integrate tuned models
    supervised_models, unsupervised_models = get_all_models(include_stacking=False)
    
    # Override baseline models with tuned versions where available
    for name, tuned_est in tuned_models.items():
        if name in supervised_models:
            supervised_models[name] = tuned_est
            
    # Build dynamic Stacking Classifier using tuned estimators
    stacking_base_estimators = [
        ('rf', tuned_models.get("Random_Forest", supervised_models["Random_Forest"])),
        ('xgb', tuned_models.get("XGBoost", supervised_models["XGBoost"])),
        ('lgb', tuned_models.get("LightGBM", supervised_models["LightGBM"])),
        ('cat', tuned_models.get("CatBoost", supervised_models["CatBoost"])),
        ('lr', supervised_models["Logistic_Regression_L2"])
    ]
    supervised_models["Stacking_Classifier"] = build_stacking_classifier(stacking_base_estimators)
    
    # 6. Benchmark Supervised Models
    print(f"\n[Step 5/6] Benchmarking {len(supervised_models)} Supervised Architectures...")
    benchmark_df, fitted_models = evaluate_supervised_models(
        supervised_models, X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_NAMES
    )
    
    # Generate Calibration Curves & Slice-Based Demographic Evaluation
    print("\n[Clinical Validation] Generating Calibration Curves & Slice Analysis...")
    generate_calibration_curves(fitted_models, X_test_scaled, y_test)
    evaluate_demographic_slices(fitted_models, X_test, X_test_scaled, y_test)
    
    # Unsupervised Clustering Evaluation
    unsup_df = evaluate_unsupervised_models(
        unsupervised_models, X_all_scaled, y
    )
    
    # 7. Model Explainability with SHAP (using Random Forest or XGBoost)
    print("\n[Step 6/6] Generating SHAP Explainability Artifacts...")
    best_model_name = "Random_Forest" if "Random_Forest" in fitted_models else "XGBoost"
    best_model = fitted_models[best_model_name]
    generate_explainability_artifacts(
        best_model, X_train_scaled, X_test_scaled, FEATURE_NAMES, X_train
    )
    
    # Summary
    print("\n" + "=" * 75)
    print(" TRAINING, TUNING & BENCHMARKING COMPLETE")
    print("=" * 75)
    print("\nTop 6 Supervised Models (including Clinical Calibration & 95% CIs):")
    summary_cols = ["Algorithm", "Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "ROC-AUC 95% CI", "Brier Score", "MCC"]
    print(benchmark_df[summary_cols].head(6).to_string(index=False))
    print("\nUnsupervised Clustering Summary:")
    print(unsup_df.to_string(index=False))
    print("\nAll model binaries saved to: models/")
    print("All evaluation charts saved to: results/")
    print("=" * 75)

if __name__ == "__main__":
    main()

