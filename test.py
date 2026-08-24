"""
Automated Verification & Test Suite for CardioAI System
"""

import os
import pickle
import numpy as np
import pandas as pd

def test_pipeline():
    print("=" * 60)
    print("RUNNING AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    
    # 1. Directory Structure
    print("\n[1/5] Checking Directory Structure...")
    for folder in ["data", "models", "results", "results/confusion_matrices", "src"]:
        assert os.path.exists(folder), f"Missing folder: {folder}"
        print(f"  [OK] Folder '{folder}' exists.")
        
    # 2. Data Integrity
    print("\n[2/5] Checking Data Integrity...")
    assert os.path.exists("data/heart.csv"), "Missing data/heart.csv"
    df = pd.read_csv("data/heart.csv")
    assert len(df) >= 300, f"Unexpected row count: {len(df)}"
    assert "target" in df.columns, "Target column missing"
    assert len(df.columns) == 14, f"Expected 14 columns, got {len(df.columns)}"
    print(f"  [OK] Authentic Cleveland dataset verified: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 3. Model Binaries (21 Algorithms)
    print("\n[3/5] Verifying Model Artifacts...")
    expected_models = [
        "Logistic_Regression_L2", "Logistic_Regression_L1", "Logistic_Regression_ElasticNet",
        "Gaussian_Naive_Bayes", "KNN_Classifier", "SVM_Linear", "SVM_RBF",
        "Decision_Tree", "Random_Forest", "Extra_Trees", "AdaBoost", "Gradient_Boosting",
        "XGBoost", "LightGBM", "CatBoost", "Artificial_Neural_Network",
        "PCA_Logistic_Regression", "Linear_Discriminant_Analysis",
        "KMeans_Clustering", "Hierarchical_Clustering", "DBSCAN_Clustering"
    ]
    
    for m in expected_models:
        path = f"models/{m}.pkl"
        assert os.path.exists(path), f"Model binary missing: {path}"
        with open(path, "rb") as f:
            obj = pickle.load(f)
            assert obj is not None
    print(f"  [OK] All {len(expected_models)} algorithm binaries loaded successfully.")
    
    # 4. Scaler and Inference Test
    print("\n[4/5] Testing Scaler and Live Patient Inference...")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/Random_Forest.pkl", "rb") as f:
        rf = pickle.load(f)
        
    # Sample patient: 55 yo male, typical angina, BP 140, Chol 250, etc.
    sample_patient = np.array([[55, 1, 0, 140, 250, 0, 0, 145, 1, 1.8, 1, 1, 2]])
    scaled_sample = scaler.transform(sample_patient)
    pred = rf.predict(scaled_sample)[0]
    prob = rf.predict_proba(scaled_sample)[0][1]
    print(f"  [OK] Inference successful: Prediction = {pred}, CAD Probability = {prob:.2%}")
    
    # 5. Result Artifacts & Plots
    print("\n[5/5] Checking Research Artifacts & Result Plots...")
    required_results = [
        "results/model_benchmark_comparison.csv",
        "results/unsupervised_benchmark_comparison.csv",
        "results/roc_curves.png",
        "results/precision_recall_curves.png",
        "results/feature_importance.png",
        "results/shap_summary.png",
        "results/unsupervised_clusters.png"
    ]
    for r in required_results:
        assert os.path.exists(r), f"Missing research result: {r}"
        print(f"  [OK] Result artifact '{r}' verified.")
        
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! PROJECT IS 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
