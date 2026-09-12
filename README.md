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

Evaluated on the authentic UCI Cleveland dataset using **Stratified 5-Fold Cross-Validation** and **500-Iteration Non-Parametric Bootstrap 95% Confidence Intervals**:

| Rank | Model Architecture | Paradigmatic Category | ROC-AUC (95% CI) | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score (95% CI) | Brier Score ↓ | MCC | 5-Fold CV |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **AdaBoost Classifier** | Ensemble Boosting | **97.19%** [0.924, 1.000] | **90.16%** | **90.37%** | 92.86% | **87.88%** | 0.8966 [0.800, 0.969] | 0.1708 | 0.8048 | 81.4% ± 5.5% |
| 🥈 | **Random Forest (Tuned)** | Bagged Ensemble | **96.43%** [0.917, 0.997] | **90.16%** | **90.37%** | 92.86% | **87.88%** | 0.8966 [0.800, 0.969] | 0.0980 | 0.8048 | 81.4% ± 2.8% |
| 🥉 | **Stacking Classifier** | Meta-Ensemble (RF+LGBM+Cat+LR) | **95.89%** [0.904, 0.994] | 86.89% | 87.34% | 92.86% | 81.82% | 0.8667 [0.762, 0.944] | **0.0861** | 0.7451 | 82.6% ± 3.2% |
| 4 | **LightGBM (Tuned)** | Gradient Boosting | 95.67% [0.902, 0.995] | 86.89% | 87.07% | 89.29% | 84.85% | 0.8621 [0.750, 0.952] | **0.0864** | 0.7389 | 82.2% ± 3.9% |
| 5 | **Logistic Regression (L2)** | Regularized GLM | 95.35% [0.893, 0.994] | 86.89% | 87.34% | 92.86% | 81.82% | 0.8667 [0.766, 0.949] | 0.0905 | 0.7451 | 83.0% ± 2.5% |
| 6 | **PCA + Logistic Regression** | Dimensionality Reduction | 95.24% [0.903, 0.989] | 85.25% | 85.55% | 89.29% | 81.82% | 0.8475 [0.731, 0.927] | 0.0944 | 0.7087 | 80.2% ± 2.3% |
| 7 | **Logistic Regression (ElasticNet)** | ElasticNet Regularization | 94.91% [0.887, 0.992] | 86.89% | 87.34% | 92.86% | 81.82% | 0.8667 [0.766, 0.949] | 0.0918 | 0.7451 | 82.6% ± 1.7% |
| 8 | **Gaussian Naïve Bayes** | Probabilistic Bayes | 94.91% [0.887, 0.994] | 86.89% | 87.61% | 96.43% | 78.79% | 0.8710 [0.763, 0.947] | 0.0943 | 0.7546 | 81.8% ± 3.1% |
| 9 | **Logistic Regression (L1)** | Sparse Feature Selection | 94.70% [0.885, 0.991] | 86.89% | 87.34% | 92.86% | 81.82% | 0.8667 [0.766, 0.949] | 0.0937 | 0.7451 | 82.2% ± 2.2% |
| 10 | **Extra Trees** | Extremely Randomized Trees | 94.59% [0.882, 0.990] | 85.25% | 86.09% | 96.43% | 75.76% | 0.8571 [0.757, 0.940] | 0.1024 | 0.7274 | 81.8% ± 2.0% |
| 11 | **SVM (RBF Kernel)** | Non-linear Kernel Machine | 94.48% [0.884, 0.990] | 85.25% | 85.55% | 89.29% | 81.82% | 0.8475 [0.731, 0.927] | 0.0964 | 0.7087 | **83.5% ± 2.9%** |
| 12 | **Artificial Neural Net (MLP)** | Deep Feedforward Net | 94.48% [0.878, 0.992] | 85.25% | 85.55% | 89.29% | 81.82% | 0.8475 [0.722, 0.935] | 0.1331 | 0.7087 | 75.2% ± 5.8% |
| 13 | **XGBoost (Tuned)** | Extreme Gradient Boosting | 94.48% [0.883, 0.986] | 88.52% | 88.58% | 89.29% | **87.88%** | 0.8772 [0.775, 0.960] | 0.1004 | 0.7700 | 82.2% ± 4.4% |
| 14 | **CatBoost (Tuned)** | Oblivious Decision Trees | 94.26% [0.877, 0.988] | 83.61% | 83.77% | 85.71% | 81.82% | 0.8276 [0.714, 0.928] | 0.1147 | 0.6731 | 82.2% ± 3.5% |
| 15 | **KNN Classifier ($k=5$)** | Metric Instance-based | 92.42% [0.848, 0.988] | **90.16%** | **90.91%** | **100.0%** | 81.82% | **0.9032** [0.818, 0.970] | 0.1115 | **0.8209** | 81.8% ± 2.0% |

> *Note*: Brier Score measures probabilistic calibration error ($\downarrow$ lower is better, where $0.0$ represents perfect probability calibration).

### 📈 Probabilistic Calibration & Diagnostic Reliability
Clinical risk scoring requires not only binary classification accuracy but well-calibrated probabilities. The **Stacking Classifier** demonstrated the superior clinical reliability:
* **Best Calibration Loss**: Stacking Classifier achieves a Brier Score of **0.0861**, outperforming standalone tree ensembles and neural architectures.
* **Calibration Curves**: Verified against empirical event frequencies via reliability diagrams (`results/calibration_curves.png`).

### ⚖️ Demographic Subgroup Fairness & Disparity Analysis
Subgroup stratification across biological sex and age cohorts (N=61 held-out test cohort):

| Model | Demographic Slice | Sample Size ($N$) | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | Brier Score ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Classifier** | Overall Cohort | 61 | 86.89% | 92.86% | 81.82% | 0.8667 | 0.9589 | **0.0861** |
| | Biological Sex: Female | 20 | **95.00%** | 85.71% | **100.0%** | 0.9231 | **1.0000** | **0.0425** |
| | Biological Sex: Male | 41 | 82.93% | **95.24%** | 70.00% | 0.8511 | 0.9476 | 0.1074 |
| | Age Cohort: < 55 yrs | 28 | **92.86%** | **100.0%** | 90.48% | 0.8750 | **1.0000** | **0.0499** |
| | Age Cohort: $\ge$ 55 yrs | 33 | 81.82% | 90.48% | 66.67% | 0.8636 | 0.9048 | 0.1168 |
| **Random Forest (Tuned)**| Overall Cohort | 61 | **90.16%** | 92.86% | 87.88% | 0.8966 | 0.9643 | 0.0980 |
| | Biological Sex: Female | 20 | **95.00%** | 85.71% | **100.0%** | 0.9231 | **1.0000** | 0.0626 |
| | Biological Sex: Male | 41 | 87.80% | **95.24%** | 80.00% | 0.8889 | 0.9452 | 0.1152 |
| | Age Cohort: < 55 yrs | 28 | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** | 0.0610 |
| | Age Cohort: $\ge$ 55 yrs | 33 | 81.82% | 90.48% | 66.67% | 0.8636 | 0.9087 | 0.1294 |

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
│   ├── 📄 Stacking_Classifier.pkl     <- Meta-ensemble architecture
│   ├── 📄 XGBoost.pkl
│   ├── 📄 scaler.pkl                  <- Fitted StandardScaler
│   └── 📄 shap_explainer.pkl          <- Pre-calculated SHAP TreeExplainer
│
├── 📁 results/                        <- Publication figures & empirical tables
│   ├── 📁 confusion_matrices/         <- 18 Confusion matrix heatmaps
│   ├── 📄 model_benchmark_comparison.csv
│   ├── 📄 slice_based_evaluation.csv  <- Demographic subgroup disparity metrics
│   ├── 📄 hyperparameter_tuning_summary.csv
│   ├── 📄 unsupervised_benchmark_comparison.csv
│   ├── 📄 calibration_curves.png      <- Clinical probability calibration curves
│   ├── 📄 roc_curves.png              <- 21-Model ROC comparison curve
│   ├── 📄 precision_recall_curves.png <- Multi-model Precision-Recall curve
│   ├── 📄 feature_importance.png      <- Clinical feature rankings
│   ├── 📄 shap_summary.png            <- SHAP global beeswarm plot
│   └── 📄 unsupervised_clusters.png   <- 2D PCA cluster projections
│
├── 📁 src/                            <- Core modular research library
│   ├── 📄 __init__.py
│   ├── 📄 data_loader.py              <- Ingestion & preprocessing pipeline
│   ├── 📄 models.py                   <- 21 ML algorithm architectures & hyperparameter tuning
│   └── 📄 evaluate.py                 <- Multi-metric validation, calibration & SHAP engine
│
├── 📄 app.py                          <- Streamlit clinical decision support dashboard
├── 📄 train_and_evaluate.py           <- Master training pipeline orchestrator
├── 📄 test.py                         <- Automated test verification suite
├── 📄 generate_term_paper.py          <- Formatted publication paper generator
├── 📄 TERM_PAPER_BENEDICT_BAAH.md     <- Complete research publication manuscript
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
