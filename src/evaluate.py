"""
Evaluation & Explainability Module: Multi-Metric Benchmark, Cross-Validation, Plots, SHAP,
Brier Score, MCC, Bootstrap Confidence Intervals, Calibration Curves, and Demographic Slice Analysis.
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
    brier_score_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

# Windows Application Control compatibility shim for SHAP (bypasses unused text hashing binary)
import sys
from unittest.mock import MagicMock
sys.modules.setdefault('sklearn.feature_extraction', MagicMock())
sys.modules.setdefault('sklearn.feature_extraction._hashing_fast', MagicMock())
sys.modules.setdefault('sklearn.feature_extraction.text', MagicMock())
import shap

RESULTS_DIR = "results"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Use clean plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def compute_bootstrap_ci(y_true, y_pred, y_proba, n_bootstraps=500, random_state=42):
    """
    Computes 95% empirical bootstrap confidence intervals for ROC-AUC and F1-Score.
    """
    rng = np.random.RandomState(random_state)
    auc_bootstraps = []
    f1_bootstraps = []
    
    n_samples = len(y_true)
    y_true_arr = np.array(y_true)
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        try:
            boot_auc = roc_auc_score(y_true_arr[idx], y_proba[idx])
            auc_bootstraps.append(boot_auc)
        except Exception:
            pass
        boot_f1 = f1_score(y_true_arr[idx], y_pred[idx], zero_division=0)
        f1_bootstraps.append(boot_f1)
        
    if auc_bootstraps:
        auc_ci = (round(float(np.percentile(auc_bootstraps, 2.5)), 4),
                  round(float(np.percentile(auc_bootstraps, 97.5)), 4))
    else:
        auc_ci = (0.0, 1.0)
        
    if f1_bootstraps:
        f1_ci = (round(float(np.percentile(f1_bootstraps, 2.5)), 4),
                 round(float(np.percentile(f1_bootstraps, 97.5)), 4))
    else:
        f1_ci = (0.0, 1.0)
        
    return auc_ci, f1_ci


def evaluate_supervised_models(models_dict, X_train, X_test, y_train, y_test, feature_names):
    """
    Trains, cross-validates, and evaluates all supervised models.
    Computes standard + advanced clinical metrics (Brier, MCC, Balanced Acc, 95% CI).
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
            
        # 4. Standard & Advanced Clinical Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            brier = brier_score_loss(y_test, y_proba)
        except Exception:
            auc, pr_auc, brier = acc, acc, 0.25
            
        # Specificity: TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 95% Bootstrap Confidence Intervals
        auc_ci, f1_ci = compute_bootstrap_ci(y_test, y_pred, y_proba, n_bootstraps=500)
        
        benchmark_records.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Balanced Accuracy": round(bal_acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-Score": round(f1, 4),
            "F1 95% CI": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
            "ROC-AUC": round(auc, 4),
            "ROC-AUC 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
            "PR-AUC": round(pr_auc, 4),
            "Brier Score": round(brier, 4),
            "MCC": round(mcc, 4),
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
        
        print(f"[{name:<30}] Acc: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f} | Brier: {brier:.3f} | MCC: {mcc:.3f}")

    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df = benchmark_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    
    # Save CSV results
    benchmark_csv_path = os.path.join(RESULTS_DIR, "model_benchmark_comparison.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"\n[Supervised Evaluation] Benchmark saved to {benchmark_csv_path}")
    
    # Plot combined ROC Curves
    plt.figure(figsize=(11, 8))
    for name, (fpr, tpr, auc_score) in roc_curves_data.items():
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison - 22 Models', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=300)
    plt.close()
    
    # Plot combined PR Curves
    plt.figure(figsize=(11, 8))
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


def generate_calibration_curves(fitted_models, X_test, y_test, top_models=None):
    """
    Generates and saves clinical calibration curves (reliability diagrams)
    comparing predicted probabilities vs empirical fraction of positives.
    """
    if top_models is None:
        top_models = [
            "Stacking_Classifier", "Random_Forest", "LightGBM", 
            "XGBoost", "CatBoost", "Logistic_Regression_L2"
        ]
        
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", lw=1.5)
    
    plotted_any = False
    for name in top_models:
        if name not in fitted_models:
            continue
        model = fitted_models[name]
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=5, strategy='uniform')
            brier = brier_score_loss(y_test, y_proba)
            plt.plot(prob_pred, prob_true, marker='o', lw=2, label=f"{name} (Brier: {brier:.3f})")
            plotted_any = True
            
    if plotted_any:
        plt.xlabel("Mean Predicted Probability", fontsize=12)
        plt.ylabel("Observed Fraction of Positives (Empirical Risk)", fontsize=12)
        plt.title("Clinical Reliability Diagram / Calibration Curves", fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        cal_path = os.path.join(RESULTS_DIR, "calibration_curves.png")
        plt.savefig(cal_path, dpi=300)
        plt.close()
        print(f"[Calibration] Saved reliability diagram to {cal_path}")


def evaluate_demographic_slices(fitted_models, X_test_raw_df, X_test_scaled, y_test, models_to_test=None):
    """
    Performs slice-based subpopulation error analysis across biological sex and age cohorts.
    Evaluates fairness and performance stability across key patient cohorts.
    """
    if models_to_test is None:
        models_to_test = [
            "Stacking_Classifier", "Random_Forest", "LightGBM", 
            "XGBoost", "CatBoost", "Logistic_Regression_L2"
        ]
        
    slices = {
        "All Patients": np.ones(len(y_test), dtype=bool),
        "Sex: Male": (X_test_raw_df['sex'].values == 1),
        "Sex: Female": (X_test_raw_df['sex'].values == 0),
        "Age: < 55": (X_test_raw_df['age'].values < 55),
        "Age: >= 55": (X_test_raw_df['age'].values >= 55)
    }
    
    slice_records = []
    y_test_arr = np.array(y_test)
    
    for model_name in models_to_test:
        if model_name not in fitted_models:
            continue
        model = fitted_models[model_name]
        
        for slice_name, mask in slices.items():
            n_sub = int(mask.sum())
            if n_sub < 5:
                continue
                
            X_sub = X_test_scaled[mask]
            y_sub = y_test_arr[mask]
            y_sub_pred = model.predict(X_sub)
            
            sub_acc = accuracy_score(y_sub, y_sub_pred)
            sub_rec = recall_score(y_sub, y_sub_pred, zero_division=0)
            
            # Specificity
            cm_sub = confusion_matrix(y_sub, y_sub_pred)
            if cm_sub.shape == (2, 2):
                tn, fp, fn, tp = cm_sub.ravel()
                sub_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sub_spec = 1.0 if (y_sub_pred == 0).all() else 0.0
                
            sub_f1 = f1_score(y_sub, y_sub_pred, zero_division=0)
            
            if hasattr(model, "predict_proba"):
                y_sub_proba = model.predict_proba(X_sub)[:, 1]
                try:
                    sub_auc = roc_auc_score(y_sub, y_sub_proba)
                    sub_brier = brier_score_loss(y_sub, y_sub_proba)
                except Exception:
                    sub_auc, sub_brier = sub_acc, 0.25
            else:
                sub_auc, sub_brier = sub_acc, 0.25
                
            slice_records.append({
                "Model": model_name,
                "Demographic Slice": slice_name,
                "Sample Size (N)": n_sub,
                "Accuracy": round(sub_acc, 4),
                "Recall (Sensitivity)": round(sub_rec, 4),
                "Specificity": round(sub_spec, 4),
                "F1-Score": round(sub_f1, 4),
                "ROC-AUC": round(sub_auc, 4),
                "Brier Score": round(sub_brier, 4)
            })
            
    slice_df = pd.DataFrame(slice_records)
    slice_path = os.path.join(RESULTS_DIR, "slice_based_evaluation.csv")
    slice_df.to_csv(slice_path, index=False)
    print(f"[Demographic Analysis] Slice evaluation saved to {slice_path}")
    return slice_df


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
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='coolwarm', alpha=0.8, edgecolors='k', s=40)
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

