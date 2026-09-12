# CardioAI: Machine Learning Research and Application Development

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Comparative Empirical Benchmarking of 21 Algorithmic Paradigms, Stratified 5-Fold Cross-Validation, and Explainable AI (SHAP) for Cardiovascular Risk Stratification**

---

## 🏛️ Project & Research Metadata
* **Project Title**: *Machine Learning Research and Application Development: Comparative Algorithmic Benchmarking & Explainable AI for Cardiovascular Risk Stratification*
* **Researcher / Author**: **Benedict Baah**
* **Research Supervisor**: **Dr. Timothy A Ogunleye**

---

## 📌 Executive Summary
Coronary Artery Disease (CAD) is the leading cause of global cardiovascular mortality. Early, non-invasive risk stratification can drastically reduce diagnostic latency and improve clinical triage. This research work presents **CardioAI**, an end-to-end clinical machine learning framework that:
1. Systematically benchmarks **21 algorithmic paradigms** across Generalized Linear Models ($L_1$, $L_2$, ElasticNet), Kernel Support Vector Machines, Tree-based Ensembles, Modern Gradient Boosters (XGBoost, LightGBM, CatBoost), Artificial Neural Networks (Multi-Layer Perceptrons), Dimensionality Reduction (PCA, LDA), and Unsupervised Clustering ($K$-Means, Hierarchical, DBSCAN).
2. Uses the canonical **UCI Cleveland Heart Disease Cohort** ($N=303$, 13 clinical biomarkers) with **Stratified 5-Fold Cross-Validation** to prevent data leakage and ensure robust generalization.
3. Integrates **Explainable Artificial Intelligence (XAI)** via SHAP (SHapley Additive exPlanations) to provide clinicians with transparent, feature-level risk factor attributions (waterfall force plots).
4. Deploys an interactive **Streamlit Clinical Decision Support System (CDSS)** delivering instant risk quantification, dynamic algorithm switching, and phenotypic clustering visualizations.

---

## 🏆 Empirical Benchmark Leaderboard (21 Algorithmic Paradigms)

Evaluated on the authentic UCI Cleveland dataset using **Stratified 5-Fold Cross-Validation**:

| Rank | Model Architecture | Paradigmatic Category | ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | 5-Fold CV Mean |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **AdaBoost Classifier** | Ensemble Boosting | **97.19%** | **90.16%** | 92.86% | 87.88% | 0.8966 | 81.4% ± 5.5% |
| 🥈 | **LightGBM Classifier** | Gradient Boosting | **95.56%** | 88.52% | 92.86% | 84.85% | 0.8814 | 78.5% ± 3.5% |
| 🥉 | **Logistic Regression (L2)** | Regularized GLM | **95.35%** | 86.89% | 92.86% | 81.82% | 0.8667 | 83.0% ± 2.5% |
| 4 | **PCA + Logistic Regression** | Dimensionality Reduction | **95.24%** | 85.25% | 89.29% | 81.82% | 0.8475 | 80.2% ± 2.3% |
| 5 | **CatBoost Classifier** | Oblivious Decision Trees | **95.13%** | **90.16%** | 92.86% | 87.88% | 0.8966 | 80.6% ± 3.7% |
| 6 | **Random Forest Classifier** | Bagged Ensemble | **95.00%** | **90.16%** | **96.43%** | 84.85% | **0.9000** | 78.9% ± 2.5% |
| 7 | **Gaussian Naïve Bayes** | Probabilistic Bayes | **94.90%** | 86.89% | **96.43%** | 78.79% | 0.8710 | 81.8% ± 3.1% |
| 8 | **SVM (RBF Kernel)** | Non-linear Kernel Machine | **94.70%** | 85.25% | 89.29% | 81.82% | 0.8475 | 82.6% ± 3.1% |
| 9 | **Artificial Neural Network (MLP)** | Deep Feedforward Net | **94.50%** | 85.25% | 89.29% | 81.82% | 0.8475 | 75.2% ± 5.8% |
| 10 | **XGBoost Classifier** | Extreme Gradient Boosting | **94.00%** | 85.25% | 92.86% | 78.79% | 0.8525 | 77.7% ± 3.6% |
| 11 | **Linear Discriminant Analysis (LDA)** | Fisher Discriminant | 93.94% | 83.61% | 85.71% | 81.82% | 0.8276 | **83.5% ± 2.7%** |
| 12 | **KNN Classifier ($k=5$)** | Metric Instance-based | 92.42% | **90.16%** | **100.0%** | 81.82% | **0.9032** | 81.8% ± 2.0% |

### Unsupervised Latent Patient Phenotyping
| Algorithm | Estimated Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz | Adjusted Rand Index (vs Truth) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$K$-Means Clustering** | 2 | **0.1759** | **2.0919** | **62.45** | **0.4293** |
| **Hierarchical Agglomerative** | 2 | 0.1293 | 2.5925 | 42.99 | 0.1785 |
| **DBSCAN** | 1 (Core) | -1.0000 | -1.0000 | -1.00 | 0.0000 |

---

## 🔬 Clinical Feature Space (UCI Cleveland Cohort)
| Feature Name | Clinical Description | Range / Categories |
| :--- | :--- | :--- |
| `age` | Patient chronological age | 29 - 77 years |
| `sex` | Biological sex | $0 = \text{Female}, 1 = \text{Male}$ |
| `cp` | Chest pain classification | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| `trestbps` | Resting systolic blood pressure | 94 - 200 mm Hg on hospital admission |
| `chol` | Serum total cholesterol | 126 - 564 mg/dl |
| `fbs` | Fasting blood glucose | $1 = \text{Fasting Blood Sugar} > 120\text{ mg/dl}, 0 = \text{Normal}$ |
| `restecg` | Resting electrocardiographic findings | 0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy |
| `thalach` | Peak exercise heart rate achieved | 71 - 202 bpm |
| `exang` | Exercise-induced angina | $1 = \text{Yes}, 0 = \text{No}$ |
| `oldpeak` | Exercise-induced ST segment depression | 0.0 - 6.2 mm relative to baseline |
| `slope` | Peak exercise ST segment slope | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Major coronary vessels fluoroscopy | 0 - 3 major vessels highlighted |
| `thal` | Thallium-201 myocardial scintigraphy | 0: Normal Perfusion, 1: Fixed Defect, 2: Reversible Defect |
| `target` | Angiographic Coronary Artery Disease | $0 = \text{Absence } (<50\% \text{ stenosis}), 1 = \text{Presence } (>50\% \text{ stenosis})$ |

---

## 🚀 Execution & Quick Start Guide

### 1. Environment Setup
```bash
git clone https://github.com/BenedictBen/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

### 2. Run Master Training & Benchmark Pipeline
```bash
python train_and_evaluate.py
```

### 3. Run Automated Validation Test Suite
```bash
python test.py
```

### 4. Launch the Clinical Decision Support Web App
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501`.

---

## 📁 Research Project Structure
```text
heart-disease-prediction/
│
├── 📁 data/                           <- Canonical dataset repository
│   └── 📄 heart.csv                   <- Authentic UCI Cleveland dataset (303 records)
│
├── 📁 models/                         <- Serialized model binaries & inference artifacts
│   ├── 📄 AdaBoost.pkl
│   ├── 📄 Artificial_Neural_Network.pkl
│   ├── 📄 CatBoost.pkl
│   ├── 📄 LightGBM.pkl
│   ├── 📄 Random_Forest.pkl
│   ├── 📄 XGBoost.pkl
│   ├── 📄 scaler.pkl                  <- Fitted StandardScaler
│   └── 📄 shap_explainer.pkl          <- Pre-calculated SHAP TreeExplainer
│
├── 📁 results/                        <- Publication figures & empirical tables
│   ├── 📁 confusion_matrices/         <- 18 Confusion matrix heatmaps
│   ├── 📄 model_benchmark_comparison.csv
│   ├── 📄 unsupervised_benchmark_comparison.csv
│   ├── 📄 roc_curves.png              <- 21-Model ROC comparison curve
│   ├── 📄 precision_recall_curves.png <- Multi-model Precision-Recall curve
│   ├── 📄 feature_importance.png      <- Clinical feature rankings
│   ├── 📄 shap_summary.png            <- SHAP global beeswarm plot
│   └── 📄 unsupervised_clusters.png   <- 2D PCA cluster projections
│
├── 📁 src/                            <- Core modular research library
│   ├── 📄 __init__.py
│   ├── 📄 data_loader.py              <- Ingestion & preprocessing pipeline
│   ├── 📄 models.py                   <- 21 ML algorithm architectures
│   └── 📄 evaluate.py                 <- Multi-metric validation & SHAP engine
│
├── 📄 app.py                          <- Streamlit clinical decision support dashboard
├── 📄 train_and_evaluate.py           <- Master training pipeline orchestrator
├── 📄 test.py                         <- Automated test verification suite
├── 📄 requirements.txt                <- Package dependencies
└── 📄 README.md                       <- Research documentation
```

---

## 📖 Citation & Academic Attribution
If you reference this research framework or dataset analysis in your academic work, please cite:

```bibtex
@article{baah2026cardioai,
  title={Machine Learning Research and Application Development: Comparative Empirical Benchmarking of 21 Algorithmic Paradigms and Explainable AI for Cardiovascular Risk Stratification},
  author={Baah, Benedict and Ogunleye, Timothy A},
  year={2026}
}
```
