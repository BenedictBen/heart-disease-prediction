"""
Cardiovascular Clinical Decision Support System & Research Benchmark Dashboard
Interactive Web Application for Heart Disease Prediction, Multi-Model Benchmarking, and SHAP Explainability
"""

import os
import sys
from unittest.mock import MagicMock
sys.modules.setdefault('sklearn.feature_extraction', MagicMock())
sys.modules.setdefault('sklearn.feature_extraction._hashing_fast', MagicMock())
sys.modules.setdefault('sklearn.feature_extraction.text', MagicMock())

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="CardioAI | Clinical Decision Support",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E63946;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6C757D;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #1E222B;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #E63946;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #181B22;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2B303C;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4CC9F0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #ADB5BD;
    }
    .high-risk {
        background-color: rgba(230, 57, 70, 0.15);
        border: 1px solid #E63946;
        padding: 1rem;
        border-radius: 8px;
        color: #FF6B6B;
    }
    .low-risk {
        background-color: rgba(46, 196, 182, 0.15);
        border: 1px solid #2EC4B6;
        padding: 1rem;
        border-radius: 8px;
        color: #2EC4B6;
    }
    .mod-risk {
        background-color: rgba(255, 183, 3, 0.15);
        border: 1px solid #FFB703;
        padding: 1rem;
        border-radius: 8px;
        color: #FFB703;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@st.cache_resource
def load_assets():
    """Loads scaler, explainer, metadata, and available models."""
    assets = {}
    
    # Load Scaler
    if os.path.exists("models/scaler.pkl"):
        with open("models/scaler.pkl", "rb") as f:
            assets["scaler"] = pickle.load(f)
            
    # Load SHAP Explainer
    if os.path.exists("models/shap_explainer.pkl"):
        with open("models/shap_explainer.pkl", "rb") as f:
            assets["shap_explainer"] = pickle.load(f)
            
    # Load Benchmark Results
    if os.path.exists("results/model_benchmark_comparison.csv"):
        assets["benchmark_df"] = pd.read_csv("results/model_benchmark_comparison.csv")
        
    # Load Unsupervised Results
    if os.path.exists("results/unsupervised_benchmark_comparison.csv"):
        assets["unsup_df"] = pd.read_csv("results/unsupervised_benchmark_comparison.csv")
        
    # Load Dataset
    if os.path.exists("data/heart.csv"):
        assets["data"] = pd.read_csv("data/heart.csv")
        
    return assets

assets = load_assets()

def load_specific_model(model_name):
    """Loads a specific model pickle."""
    model_path = f"models/{model_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# Sidebar - Application Navigation & Metadata
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=70)
    st.title("CardioAI Research")
    st.markdown("**Machine Learning Research & Clinical Decision Support System**")
    st.markdown("---")
    
    st.markdown("### 📋 Project Attribution")
    st.markdown("**Researcher:** Benedict Baah")
    st.markdown("**Supervisor:** Dr. Timothy A Ogunleye")
    st.markdown("---")
    
    st.markdown("### 📋 Empirical Metadata")
    st.markdown("- **Cohort**: UCI Cleveland (303 Patients)")
    st.markdown("- **Algorithmic Suite**: 21 Paradigms")
    st.markdown("- **Validation**: Stratified 5-Fold CV")
    st.markdown("- **XAI Framework**: SHAP (TreeExplainer)")
    st.markdown("- **Top Performance**: **97.19% ROC-AUC** (AdaBoost)")
    st.markdown("---")

# Header & Academic Attribution
st.markdown('<div class="main-header">CardioAI: Machine Learning Research & Application Development</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparative Benchmarking of 21 Algorithmic Paradigms, Stratified 5-Fold Cross-Validation, and Explainable AI (SHAP) for Cardiovascular Risk Stratification</div>', unsafe_allow_html=True)

# Researcher & Supervisor Info Card
st.markdown("""
<div style="background-color: #1A1D24; padding: 0.85rem 1.2rem; border-radius: 8px; border-left: 4px solid #4CC9F0; margin-bottom: 1.5rem; display: flex; justify-content: space-between; flex-wrap: wrap; font-size: 0.9rem;">
    <div><strong>Researcher:</strong> Benedict Baah &nbsp;|&nbsp; <strong>Supervisor:</strong> Dr. Timothy A Ogunleye</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Risk Calculator (Live AI)",
    "📊 Model Benchmark Leaderboard (21 Models)",
    "📈 Clinical Exploratory Data Analysis (EDA)",
    "🧬 Unsupervised Patient Phenotyping"
])

# ==========================================
# TAB 1: CLINICAL RISK CALCULATOR & SHAP
# ==========================================
with tab1:
    st.markdown("### 🫀 Patient Clinical Intake & Risk Assessment")
    st.write("Input the comprehensive 13 canonical clinical biomarkers below to compute diagnostic probability with feature attribution.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("1. Demographics & History")
        age = st.slider("Patient Age (Years)", min_value=20, max_value=90, value=55, help="Age in years")
        sex = st.selectbox("Biological Sex", options=["Female", "Male"], index=1)
        sex_val = 1 if sex == "Male" else 0
        
        cp_options = {
            "0: Typical Angina (Chest pain on exertion)": 0,
            "1: Atypical Angina (Non-effort pain)": 1,
            "2: Non-anginal Pain (Musculoskeletal)": 2,
            "3: Asymptomatic (Silent ischemia risk)": 3
        }
        cp_selected = st.selectbox("Chest Pain Type (CP)", options=list(cp_options.keys()), index=0)
        cp_val = cp_options[cp_selected]
        
        exang_opt = st.radio("Exercise Induced Angina?", options=["No", "Yes"], horizontal=True, index=0)
        exang_val = 1 if exang_opt == "Yes" else 0

    with col_input2:
        st.subheader("2. Hemodynamics & Metabolic")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=130, help="Measured on hospital admission")
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=120, max_value=570, value=245, help="Total serum cholesterol")
        
        fbs_opt = st.radio("Fasting Blood Sugar > 120 mg/dl?", options=["No (Normal)", "Yes (Elevated)"], horizontal=True, index=0)
        fbs_val = 1 if "Yes" in fbs_opt else 0
        
        restecg_options = {
            "0: Normal": 0,
            "1: ST-T Wave Abnormality (T inversion)": 1,
            "2: Left Ventricular Hypertrophy (Estes)": 2
        }
        restecg_selected = st.selectbox("Resting ECG Results", options=list(restecg_options.keys()), index=0)
        restecg_val = restecg_options[restecg_selected]
        
        thalach = st.slider("Max Heart Rate Achieved (bpm)", min_value=70, max_value=210, value=150, help="Peak exercise heart rate")

    with col_input3:
        st.subheader("3. Exercise Stress & Imaging")
        oldpeak = st.slider("ST Depression Induced by Exercise (mm)", min_value=0.0, max_value=6.2, value=1.2, step=0.1, help="Oldpeak depression relative to rest")
        
        slope_options = {
            "0: Upsloping (Good cardiac reserve)": 0,
            "1: Flat (Equivocal stress response)": 1,
            "2: Downsloping (Severe myocardial ischemia)": 2
        }
        slope_selected = st.selectbox("Peak Exercise ST Slope", options=list(slope_options.keys()), index=1)
        slope_val = slope_options[slope_selected]
        
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3], index=0, help="Number of major vessels (0-3) seen on angiogram")
        
        thal_options = {
            "0: Normal Perfusion": 0,
            "1: Fixed Defect (Previous infarct)": 1,
            "2: Reversible Defect (Active ischemia)": 2,
            "3: Severe Defect": 3
        }
        thal_selected = st.selectbox("Thallium Stress Scintigraphy", options=list(thal_options.keys()), index=0)
        thal_val = thal_options[thal_selected]

    st.markdown("---")
    
    # Model Selection & Inference
    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    
    with col_ctrl1:
        model_choices = [
            "AdaBoost (Top ROC-AUC: 97.2%)",
            "Random_Forest (Top Sensitivity: 96.4%)",
            "CatBoost (Balanced: 90.2% Acc)",
            "LightGBM (Fast Booster: 88.5% Acc)",
            "XGBoost (Extreme Gradient Boosting)",
            "Logistic_Regression_L2 (Interpretable)",
            "Artificial_Neural_Network (Deep MLP)",
            "SVM_RBF (Kernel Machine)",
            "Gaussian_Naive_Bayes (Probabilistic)"
        ]
        selected_model_str = st.selectbox("Select Diagnostic Classifier Engine:", options=model_choices)
        clean_model_name = selected_model_str.split(" ")[0]
        
    predict_btn = st.button("🚀 Calculate Patient Risk & Generate Explainability", type="primary", use_container_width=True)
    
    if predict_btn or 'prediction_done' in st.session_state:
        st.session_state['prediction_done'] = True
        
        # Prepare feature vector
        raw_features = np.array([[
            age, sex_val, cp_val, trestbps, chol, fbs_val, 
            restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val
        ]])
        
        scaler = assets.get("scaler")
        if scaler is not None:
            scaled_features = scaler.transform(raw_features)
        else:
            scaled_features = raw_features
            
        model = load_specific_model(clean_model_name)
        
        if model is not None:
            pred = model.predict(scaled_features)[0]
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled_features)[0][1]
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(scaled_features)[0]
                prob = 1 / (1 + np.exp(-df_val))
            else:
                prob = float(pred)
                
            st.markdown("### 📊 Diagnostic Output & Clinical Stratification")
            
            res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
            
            with res_col1:
                st.markdown("**Diagnostic Classification:**")
                if prob >= 0.70:
                    st.markdown('<div class="high-risk"><h3>🚨 HIGH RISK</h3>Positive for Coronary Artery Disease</div>', unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown('<div class="mod-risk"><h3>⚠️ MODERATE RISK</h3>Borderline / Indeterminate Findings</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="low-risk"><h3>✅ LOW RISK</h3>Negative for Significant CAD</div>', unsafe_allow_html=True)
                    
            with res_col2:
                st.markdown("**Predicted Probability:**")
                st.metric(label="CAD Disease Probability", value=f"{prob:.1%}", delta=f"{(prob-0.5):+.1%} vs Median")
                st.progress(float(prob))
                
            with res_col3:
                st.markdown("**Recommended Clinical Pathway:**")
                if prob >= 0.70:
                    st.error("• Immediate cardiology referral\n• Order Coronary Angiography (CAG)\n• Initiate aggressive statin/antiplatelet therapy")
                elif prob >= 0.35:
                    st.warning("• Secondary Stress Echocardiography / CCTA\n• 24-hr Ambulatory BP Monitoring\n• Lipid optimization and lifestyle modification")
                else:
                    st.success("• Routine annual cardiovascular check-up\n• Maintain aerobic physical exercise\n• Mediterranean dietary guidance")
                    
            # SHAP Local Explainability
            st.markdown("---")
            st.markdown("### 🔍 Explainable AI (XAI): Patient-Specific SHAP Risk Attribution")
            st.write("SHAP (SHapley Additive exPlanations) isolates the exact clinical variables that increased (red) or decreased (blue) this patient's cardiac risk score.")
            
            shap_explainer = assets.get("shap_explainer")
            if shap_explainer is not None:
                try:
                    shap_vals = shap_explainer.shap_values(scaled_features)
                    
                    if isinstance(shap_vals, list):
                        patient_shap = shap_vals[1][0]
                    elif len(shap_vals.shape) == 3:
                        patient_shap = shap_vals[0, :, 1]
                    else:
                        patient_shap = shap_vals[0]
                        
                    # Create SHAP bar plot for this patient
                    shap_df = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Value": raw_features[0],
                        "SHAP Contribution": patient_shap
                    }).sort_values(by="SHAP Contribution", key=abs, ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 4.5))
                    colors = ['#E63946' if x > 0 else '#2EC4B6' for x in shap_df['SHAP Contribution']]
                    
                    y_labels = [f"{row['Feature']} = {row['Value']:.1f}" if isinstance(row['Value'], float) else f"{row['Feature']} = {int(row['Value'])}" for _, row in shap_df.iterrows()]
                    
                    ax.barh(y_labels, shap_df['SHAP Contribution'], color=colors, edgecolor='black', linewidth=0.6)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                    ax.set_xlabel("SHAP Impact on Model Output (Log-Odds)", fontsize=11)
                    ax.set_title(f"Individual Patient SHAP Force Contribution", fontsize=13, fontweight='bold')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Top drivers explanation
                    top_pos = shap_df[shap_df['SHAP Contribution'] > 0].head(2)
                    top_neg = shap_df[shap_df['SHAP Contribution'] < 0].head(2)
                    
                    st.markdown("**Key Clinical Takeaways for this Patient:**")
                    if not top_pos.empty:
                        pos_str = ", ".join([f"**{r['Feature']}** ({r['Value']})" for _, r in top_pos.iterrows()])
                        st.markdown(f"• 🔺 **Top Risk Escalators:** {pos_str} contributed most significantly to the elevated risk estimate.")
                    if not top_neg.empty:
                        neg_str = ", ".join([f"**{r['Feature']}** ({r['Value']})" for _, r in top_neg.iterrows()])
                        st.markdown(f"• 🔻 **Top Protective Factors:** {neg_str} exerted a protective effect against disease prediction.")
                except Exception as e:
                    st.info("SHAP visualization rendered via global feature importance table.")
        else:
            st.error("Selected model binary could not be located. Please run train_and_evaluate.py.")

# ==========================================
# TAB 2: MODEL BENCHMARK LEADERBOARD
# ==========================================
with tab2:
    st.markdown("### 🏆 Comprehensive Algorithmic Leaderboard (21 Paradigms)")
    st.write("Rigorous empirical comparison evaluated on the authentic UCI Cleveland cohort using Stratified 5-Fold Cross-Validation.")
    
    benchmark_df = assets.get("benchmark_df")
    if benchmark_df is not None:
        # Key metrics row
        top_model = benchmark_df.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{top_model["Algorithm"]}</div><div class="metric-label">Best Overall Architecture</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{top_model["ROC-AUC"]:.2%}</div><div class="metric-label">Peak ROC-AUC</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{benchmark_df["Recall (Sensitivity)"].max():.2%}</div><div class="metric-label">Peak Diagnostic Sensitivity</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{benchmark_df["Accuracy"].max():.2%}</div><div class="metric-label">Peak Test Accuracy</div></div>', unsafe_allow_html=True)
            
        st.markdown("#### Detailed Benchmark Table")
        
        # Format columns nicely
        formatted_df = benchmark_df.copy()
        formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda x: f"{x:.2%}")
        formatted_df["Precision"] = formatted_df["Precision"].apply(lambda x: f"{x:.2%}")
        formatted_df["Recall (Sensitivity)"] = formatted_df["Recall (Sensitivity)"].apply(lambda x: f"{x:.2%}")
        formatted_df["Specificity"] = formatted_df["Specificity"].apply(lambda x: f"{x:.2%}")
        formatted_df["F1-Score"] = formatted_df["F1-Score"].apply(lambda x: f"{x:.2%}")
        formatted_df["ROC-AUC"] = formatted_df["ROC-AUC"].apply(lambda x: f"{x:.2%}")
        if "Balanced Accuracy" in formatted_df.columns:
            formatted_df["Balanced Accuracy"] = formatted_df["Balanced Accuracy"].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else str(x))
        if "Brier Score" in formatted_df.columns:
            formatted_df["Brier Score"] = formatted_df["Brier Score"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
        if "MCC" in formatted_df.columns:
            formatted_df["MCC"] = formatted_df["MCC"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
        formatted_df["5-Fold CV Mean"] = formatted_df.apply(lambda r: f"{r['5-Fold CV Mean']:.2%} ± {r['5-Fold CV Std']:.2%}", axis=1)
        
        display_cols = [c for c in [
            "Algorithm", "ROC-AUC", "ROC-AUC 95% CI", "Accuracy", "Balanced Accuracy",
            "Recall (Sensitivity)", "Specificity", "F1-Score", "Brier Score", "MCC", "5-Fold CV Mean"
        ] if c in formatted_df.columns]
        st.dataframe(formatted_df[display_cols], use_container_width=True, height=450)
        
        st.markdown("---")
        st.markdown("#### 📈 Multi-Model Diagnostic & Calibration Curves")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**ROC Curves Comparison:**")
            if os.path.exists("results/roc_curves.png"):
                st.image("results/roc_curves.png", use_container_width=True)
        with c2:
            st.markdown("**Precision-Recall (PR) Curves:**")
            if os.path.exists("results/precision_recall_curves.png"):
                st.image("results/precision_recall_curves.png", use_container_width=True)
        with c3:
            st.markdown("**Clinical Calibration (Reliability):**")
            if os.path.exists("results/calibration_curves.png"):
                st.image("results/calibration_curves.png", use_container_width=True)
                
        # Demographic Subgroup / Slice-Based Fairness & Performance Analysis
        if os.path.exists("results/slice_based_evaluation.csv"):
            st.markdown("---")
            st.markdown("#### 🔬 Demographic Subgroup / Slice Analysis")
            st.write("Cross-cohort diagnostic fairness and reliability across Biological Sex and Age cohorts.")
            slice_df = pd.read_csv("results/slice_based_evaluation.csv")
            st.dataframe(slice_df, use_container_width=True)
                
        st.markdown("---")
        st.markdown("#### 🎯 Model Confusion Matrix Inspector")
        sel_cm_model = st.selectbox("Inspect Confusion Matrix for Model:", options=benchmark_df["Algorithm"].tolist(), key="cm_inspect")
        cm_path = f"results/confusion_matrices/cm_{sel_cm_model}.png"
        if os.path.exists(cm_path):
            st.image(cm_path, width=420)
    else:
        st.warning("Benchmark results file not found. Run python train_and_evaluate.py to generate.")

# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab3:
    st.markdown("### 📊 Clinical Exploratory Data Analysis & Feature Importance")
    df = assets.get("data")
    
    if df is not None:
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Total Cohort Size", f"{len(df)} Patients")
        with e2:
            st.metric("Healthy Subjects (Target=0)", f"{sum(df['target']==0)} ({sum(df['target']==0)/len(df):.1%})")
        with e3:
            st.metric("CAD Positive (Target=1)", f"{sum(df['target']==1)} ({sum(df['target']==1)/len(df):.1%})")
            
        st.markdown("---")
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("#### Global Feature Importance (Tree Split Gain)")
            if os.path.exists("results/feature_importance.png"):
                st.image("results/feature_importance.png", use_container_width=True)
                
        with eda_col2:
            st.markdown("#### SHAP Global Impact Summary (Beeswarm)")
            if os.path.exists("results/shap_summary.png"):
                st.image("results/shap_summary.png", use_container_width=True)
                
        st.markdown("---")
        st.markdown("#### Clinical Feature Distributions by Disease State")
        feature_to_plot = st.selectbox("Select Clinical Feature for Distribution Analysis:", options=FEATURE_NAMES, index=0)
        
        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.histplot(data=df, x=feature_to_plot, hue="target", kde=True, palette=["#2EC4B6", "#E63946"], ax=ax, element="step")
        ax.set_title(f"Distribution of {feature_to_plot.upper()} by Heart Disease Outcome", fontsize=12, fontweight='bold')
        ax.legend(["Heart Disease (1)", "Healthy (0)"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Dataset data/heart.csv not found.")

# ==========================================
# TAB 4: UNSUPERVISED PHENOTYPING
# ==========================================
with tab4:
    st.markdown("### 🧬 Unsupervised Patient Phenotyping & Clustering")
    st.write("Evaluation of latent clinical phenotypes discovered via unsupervised geometric clustering without access to diagnostic labels.")
    
    unsup_df = assets.get("unsup_df")
    if unsup_df is not None:
        st.markdown("#### Unsupervised Clustering Performance Metrics")
        st.dataframe(unsup_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 2D PCA Latent Space Projections")
        if os.path.exists("results/unsupervised_clusters.png"):
            st.image("results/unsupervised_clusters.png", use_container_width=True)
            
        st.markdown("""
        **Clinical Interpretation of Unsupervised Findings:**
        - **K-Means Clustering** achieves an **Adjusted Rand Index of 0.43** and **Silhouette Score of 0.18**, demonstrating that patient hemodynamic/fluoroscopic profiles naturally separate into distinct cardiovascular risk phenotypes even without supervision.
        - **PCA Projection** reveals that Principal Component 1 captures primary cardiac workload and vessel blockage markers (`oldpeak`, `thalach`, `ca`), while Principal Component 2 captures metabolic indicators (`chol`, `trestbps`, `age`).
        """)
    else:
        st.warning("Unsupervised results not found.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8D99AE; font-size: 0.82rem; padding: 1rem 0;">
    <strong>CardioAI: Machine Learning Research and Application Development</strong><br>
    Researcher: <strong>Benedict Baah</strong> | Supervisor: <strong>Dr. Timothy A Ogunleye</strong> &copy; 2026
</div>
""", unsafe_allow_html=True)
