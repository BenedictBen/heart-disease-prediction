"""
Evaluation & Explainability Module: Multi-Metric Benchmark, Cross-Validation, Plots, and SHAP
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA
import shap

RESULTS_DIR = "results"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Use clean plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def evaluate_supervised_models(models_dict, X_train, X_test, y_train, y_test, feature_names):
    """
    Trains, cross-validates, and evaluates all supervised models.
    Returns a benchmark DataFrame, fitted models dictionary, and generates ROC/PR/CM plots.
    """
    benchmark_records = []
    fitted_models = {}
    roc_curves_data = {}
    pr_curves_data = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models_dict.items():
        # 1. 5-Fold Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_roc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # 2. Fit model on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Save individual model pickle
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
            
        # 3. Test set predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
            y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
        else:
            y_proba = y_pred.astype(float)
            
        # 4. Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        except Exception:
            auc = acc
            pr_auc = acc
            
        # Specificity: TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        benchmark_records.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "5-Fold CV Mean": round(cv_scores.mean(), 4),
            "5-Fold CV Std": round(cv_scores.std(), 4),
            "CV ROC-AUC Mean": round(cv_roc_scores.mean(), 4)
        })
        
        # Store curve data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr, tpr, auc)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_curves_data[name] = (recall_vals, precision_vals, pr_auc)
        
        # Save individual confusion matrix plot
        plt.figure(figsize=(4.5, 3.5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Healthy', 'Heart Disease'],
                    yticklabels=['Healthy', 'Heart Disease'])
        plt.title(f"{name}\nAcc: {acc:.2%} | Rec: {rec:.2%}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(CM_DIR, f"cm_{name}.png"), dpi=200)
        plt.close()
        
        print(f"[{name:<30}] Acc: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df = benchmark_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    
    # Save CSV results
    benchmark_csv_path = os.path.join(RESULTS_DIR, "model_benchmark_comparison.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"\n[Supervised Evaluation] Benchmark saved to {benchmark_csv_path}")
    
    # Plot combined ROC Curves
    plt.figure(figsize=(10, 7))
    for name, (fpr, tpr, auc_score) in roc_curves_data.items():
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison - 18+ Models', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=300)
    plt.close()
    
    # Plot combined PR Curves
    plt.figure(figsize=(10, 7))
    for name, (rec_vals, prec_vals, pr_score) in pr_curves_data.items():
        plt.plot(rec_vals, prec_vals, lw=1.8, label=f"{name} (PR-AUC = {pr_score:.2f})")
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall (PR) Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curves.png"), dpi=300)
    plt.close()
    
    return benchmark_df, fitted_models

def evaluate_unsupervised_models(unsupervised_dict, X_scaled, y_true):
    """
    Evaluates clustering algorithms on unsupervised geometry and against clinical ground truth.
    """
    unsupervised_records = []
    
    # 2D PCA for visual projection
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(14, 4))
    
    # Ground Truth plot
    plt.subplot(1, len(unsupervised_dict) + 1, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='coolwarm', alpha=0.8, edgecolors='k', s=40)
    plt.title("Clinical Ground Truth\n(0=Healthy, 1=Disease)", fontsize=10, fontweight='bold')
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    
    idx = 2
    for name, model in unsupervised_dict.items():
        cluster_labels = model.fit_predict(X_scaled)
        
        # Filter noise for DBSCAN (-1)
        valid_mask = cluster_labels != -1
        n_clusters = len(set(cluster_labels[valid_mask]))
        
        if n_clusters > 1 and valid_mask.sum() > 5:
            sil = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask])
            db = davies_bouldin_score(X_scaled[valid_mask], cluster_labels[valid_mask])
            ch = calinski_harabasz_score(X_scaled[valid_mask], cluster_labels[valid_mask])
            ari = adjusted_rand_score(y_true, cluster_labels)
        else:
            sil, db, ch, ari = -1, -1, -1, 0
            
        unsupervised_records.append({
            "Clustering Algorithm": name,
            "Estimated Clusters": n_clusters,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin Index": round(db, 4),
            "Calinski-Harabasz Index": round(ch, 4),
            "Adjusted Rand Index (vs Truth)": round(ari, 4)
        })
        
        # Save model pickle
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
            
        plt.subplot(1, len(unsupervised_dict) + 1, idx)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8, edgecolors='k', s=40)
        plt.title(f"{name}\nClusters: {n_clusters} | Sil: {sil:.2f}", fontsize=10, fontweight='bold')
        plt.xlabel("PCA 1")
        idx += 1
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unsupervised_clusters.png"), dpi=300)
    plt.close()
    
    unsup_df = pd.DataFrame(unsupervised_records)
    unsup_csv_path = os.path.join(RESULTS_DIR, "unsupervised_benchmark_comparison.csv")
    unsup_df.to_csv(unsup_csv_path, index=False)
    print(f"[Unsupervised Evaluation] Saved cluster metrics to {unsup_csv_path}")
    return unsup_df

def generate_explainability_artifacts(best_tree_model, X_train_scaled, X_test_scaled, feature_names, X_train_df):
    """
    Computes global feature importances, SHAP values, and serializes the SHAP TreeExplainer
    so the Streamlit frontend can produce real-time patient risk waterfall plots.
    """
    print("\n[Explainability] Computing SHAP values and Feature Importance...")
    
    # 1. Standard Gini / Feature Importance (if tree model)
    if hasattr(best_tree_model, "feature_importances_"):
        fi = best_tree_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": fi
        }).sort_values("Importance", ascending=False)
        fi_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
        plt.title(f"Clinical Feature Importance ({type(best_tree_model).__name__})", fontsize=13, fontweight='bold')
        plt.xlabel("Relative Importance (Gini / Split Gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
        
    # 2. SHAP TreeExplainer
    try:
        explainer = shap.TreeExplainer(best_tree_model)
        # Background summary sample for app
        background_data = X_train_scaled[:100]
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Handle binary classification 2D or 3D shap values
        if isinstance(shap_values, list):
            sv_to_plot = shap_values[1] # positive class
        elif len(shap_values.shape) == 3:
            sv_to_plot = shap_values[:, :, 1]
        else:
            sv_to_plot = shap_values
            
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv_to_plot, X_test_scaled, feature_names=feature_names, show=False)
        plt.title("SHAP Global Clinical Feature Impact (Beeswarm)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=300)
        plt.close()
        
        # Save explainer for real-time frontend use
        with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), "wb") as f:
            pickle.dump(explainer, f)
        with open(os.path.join(MODELS_DIR, "shap_background.pkl"), "wb") as f:
            pickle.dump(background_data, f)
        print("[Explainability] SHAP TreeExplainer saved to models/shap_explainer.pkl")
    except Exception as e:
        print(f"[Explainability] Warning in SHAP computation: {e}")
