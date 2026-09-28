import os
import subprocess
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_footer_page_number(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def create_base_document():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
        # Footer with centered page number
        footer = s.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        f_run = f_p.add_run()
        f_run.font.name = 'Times New Roman'
        f_run.font.size = Pt(10)
        f_run.font.color.rgb = RGBColor(100, 100, 100)
        add_footer_page_number(f_run)

    # Configure Normal Style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(30, 30, 30)
    return doc

def add_header_banner(doc, title_text, subtitle_text):
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(0)
    p_uni.paragraph_format.space_after = Pt(2)
    r_uni = p_uni.add_run("JAIN (Deemed-to-be University)")
    r_uni.font.name = 'Times New Roman'
    r_uni.font.size = Pt(16)
    r_uni.bold = True
    r_uni.font.color.rgb = RGBColor(0, 32, 96) # Academic Navy
    
    p_cdoe = doc.add_paragraph()
    p_cdoe.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cdoe.paragraph_format.space_before = Pt(0)
    p_cdoe.paragraph_format.space_after = Pt(12)
    r_cdoe = p_cdoe.add_run("Centre for Distance and Online Education / JAIN Online")
    r_cdoe.font.name = 'Times New Roman'
    r_cdoe.font.size = Pt(11)
    r_cdoe.font.color.rgb = RGBColor(80, 80, 80)
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(2)
    r_title = p_title.add_run(title_text)
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(15)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)
    
    if subtitle_text:
        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_sub.paragraph_format.space_before = Pt(0)
        p_sub.paragraph_format.space_after = Pt(14)
        r_sub = p_sub.add_run(subtitle_text)
        r_sub.font.name = 'Times New Roman'
        r_sub.font.size = Pt(11)
        r_sub.font.italic = True
        r_sub.font.color.rgb = RGBColor(100, 116, 139)

def add_meta_table(doc):
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    col_widths = [Inches(2.2), Inches(4.3)]
    metadata = [
        ("Name of Learner", "Benedict Baah"),
        ("USN (Student ID)", "[Insert Your USN / Student ID]"),
        ("Elective / Specialization", "Computer Science & IT (CSIT) / Data Analytics (DAAN)"),
        ("Faculty Supervisors", "Dr. Kavitha R G & Dr. Kwasi Kwateng"),
        ("Live Web Application", "https://heart-disease-prediction-hv6rhgtwcbaogahkc2mgne.streamlit.app/"),
        ("GitHub Repository", "https://github.com/BenedictBen/heart-disease-prediction")
    ]
    
    for row_idx, (k, v) in enumerate(metadata):
        row = table.rows[row_idx]
        cell_k = row.cells[0]
        cell_v = row.cells[1]
        
        cell_k.width = col_widths[0]
        cell_v.width = col_widths[1]
        set_cell_margins(cell_k, 80, 80, 120, 120)
        set_cell_margins(cell_v, 80, 80, 120, 120)
        set_cell_background(cell_k, "F1F5F9")
        set_cell_background(cell_v, "FFFFFF")
        
        p_k = cell_k.paragraphs[0]
        p_k.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_k.paragraph_format.space_after = Pt(0)
        r_k = p_k.add_run(k)
        r_k.font.name = 'Times New Roman'
        r_k.font.size = Pt(11)
        r_k.bold = True
        
        p_v = cell_v.paragraphs[0]
        p_v.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_v.paragraph_format.space_after = Pt(0)
        r_v = p_v.add_run(v)
        r_v.font.name = 'Times New Roman'
        r_v.font.size = Pt(11)
        if "USN" in k:
            r_v.italic = True
            r_v.font.color.rgb = RGBColor(180, 83, 9)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 32, 96)
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12.5)
    r.bold = True
    r.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_body_paragraph(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    return p

def build_synopsis():
    doc = create_base_document()
    add_header_banner(doc, "PROJECT SYNOPSIS", "Prescribed Template as per University Project Guidelines (Annexure 2)")
    add_meta_table(doc)
    
    # Title
    add_heading_1(doc, "• Title")
    add_body_paragraph(doc, "CardioAI: Comparative Empirical Benchmarking of 21 Algorithmic Paradigms, Probabilistic Calibration, and Explainable AI for Cardiovascular Risk Stratification")
    
    # Problem Statement
    add_heading_1(doc, "• Problem Statement")
    add_body_paragraph(doc, "Coronary Artery Disease (CAD) remains the primary contributor to global cardiovascular mortality, necessitating early, accurate, and non-invasive risk stratification to alleviate clinical triage backlogs and prevent fatal acute coronary events. Traditional clinical scoring indices, such as the Framingham Risk Score, rely on simplified parametric assumptions that fail to capture subtle, high-dimensional non-linear interactions among heterogeneous physiological, metabolic, and hemodynamic biomarkers.")
    add_body_paragraph(doc, "Conversely, modern machine learning applications frequently function as uncalibrated 'black-box' systems. These models exhibit two fundamental clinical shortcomings: first, their predicted probabilities are often poorly calibrated, yielding overconfident or inaccurate numerical risk estimates that undermine clinical decision thresholds; second, their internal decision logic lacks interpretability, preventing cardiologists from understanding the patient-specific physiological drivers behind a high-risk prediction.")
    add_body_paragraph(doc, "Furthermore, clinical algorithms often exhibit unexamined demographic performance disparities across patient cohorts of different biological sexes and age brackets. There is an urgent, unmet need for an empirical clinical framework that systematically benchmarks diverse algorithmic families, enforces rigorous probabilistic calibration, audits cross-demographic fairness, and integrates transparent Explainable Artificial Intelligence (XAI) within an interactive Clinical Decision Support System (CDSS) for bedside decision-making.")
    
    # Objectives
    add_heading_1(doc, "• Objectives of the Project")
    add_body_paragraph(doc, "1. Algorithmic Benchmarking & Meta-Ensemble Formulation: To train, hyperparameter-tune, and empirically benchmark 21 diverse machine learning paradigms (including Regularized Generalized Linear Models, Support Vector Machines, Bagged and Boosted Tree Ensembles, Multi-Layer Perceptrons, Dimensionality Reduction techniques, and a Stacking Meta-Classifier) using Stratified 5-Fold Cross-Validation and non-parametric bootstrap confidence intervals on canonical cardiovascular clinical biomarkers.")
    add_body_paragraph(doc, "2. Probabilistic Calibration & Demographic Fairness Auditing: To quantify and optimize probabilistic prediction reliability using Brier Score loss and clinical calibration diagrams, while conducting rigorous slice-based fairness evaluations across patient biological sex and age cohorts to ensure equitable, bias-minimized diagnostic utility.")
    add_body_paragraph(doc, "3. Explainable AI Integration & Clinical Decision Support Deployment (Learner's Contribution): To design and deploy an open-source, interactive web-based Clinical Decision Support System (CDSS) powered by SHAP (SHapley Additive exPlanations) that translates complex meta-ensemble predictions into intuitive patient-level feature attributions, enabling cardiologists to dynamically simulate biomarker interventions and verify diagnostic explanations in real time.")
    
    # Methodology
    add_heading_1(doc, "• Project Methodology")
    add_heading_2(doc, "Type of Project & Data Collection")
    add_body_paragraph(doc, "This project represents a hybrid Research-based and Application-based empirical study. It rigorously investigates comparative algorithmic behavior under clinical validation constraints and delivers a functional, user-facing software system for clinical decision support. The research utilizes Secondary Data Collection using the canonical UCI Cleveland Heart Disease Cohort (N = 303 patient records, 13 clinical biomarkers). Features encompass continuous hemodynamic parameters (resting blood pressure, serum cholesterol, maximum exercise heart rate, exercise-induced ST depression), categorical physiological markers (chest pain etiology, resting electrocardiographic abnormalities, slope of peak exercise ST segment, thallium scintigraphy defect status), and binary risk indicators (fasting blood sugar > 120 mg/dL, exercise-induced angina, biological sex, fluoroscopic major vessel opacification).")
    
    add_heading_2(doc, "Research Design & Technical Workflow")
    add_body_paragraph(doc, "Data Preprocessing & Standardization: Missing values are screened, features are normalized using Z-score standardization (StandardScaler) fitted strictly on training folds to prevent target leakage, and the binary clinical endpoint is defined as the presence of angiographic coronary artery disease (>= 50% diameter stenosis).")
    add_body_paragraph(doc, "Cross-Validation & Hyperparameter Tuning: To ensure statistical generalization, models are evaluated via Stratified 5-Fold Cross-Validation. Top candidate architectures—including Random Forest, LightGBM, XGBoost, CatBoost, Support Vector Machines, and ElasticNet—undergo automated hyperparameter optimization via RandomizedSearchCV maximizing the Area Under the Receiver Operating Characteristic (ROC-AUC).")
    add_body_paragraph(doc, "Meta-Ensemble Stacking: A multi-level StackingClassifier is formulated, combining heterogeneous base learners (Random Forest, LightGBM, CatBoost, and regularized Logistic Regression) through a regularized Logistic Regression meta-estimator that learns optimal probability blending weights.")
    add_body_paragraph(doc, "Clinical Calibration & Statistical Evaluation: Models are comprehensively assessed across a multi-dimensional metric matrix: Classification Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, Precision-Recall AUC (PR-AUC), and Matthews Correlation Coefficient (MCC). Statistical significance is established using 500-iteration non-parametric bootstrap resampling to calculate 95% Confidence Intervals. Probabilistic calibration is computed using Brier Score Loss and visualized via reliability diagrams.")
    add_body_paragraph(doc, "Demographic Slice & Fairness Analysis: Holdout performance is decomposed across demographic slices: Biological Sex (Male vs. Female) and Chronological Age (< 55 vs. >= 55 years), validating clinical stability across vulnerable groups.")
    add_body_paragraph(doc, "Explainable AI (XAI) & Deployment: SHAP TreeExplainer calculates additive Shapley values based on cooperative game theory, generating global beeswarm importance rankings and personalized waterfall force plots. The calibrated models and explainability engine are packaged into an interactive Streamlit Clinical Decision Support Dashboard (app.py), enabling clinicians to toggle models, adjust patient biomarkers, inspect calibration curves, and review patient risk attributions.")
    
    # Limitations
    add_heading_1(doc, "• Limitations")
    add_body_paragraph(doc, "While this study establishes an empirical benchmark and clinical interface, several methodological limitations must be acknowledged:")
    add_body_paragraph(doc, "1. Retrospective Single-Center Cohort: The empirical analysis relies on the Cleveland Clinic patient repository (N = 303). Although globally recognized as a clinical benchmark, the sample size is moderate and reflects single-center demographic and clinical referral patterns from the initial data gathering period.")
    add_body_paragraph(doc, "2. Exclusion of High-Dimensional Modalities: The feature space is restricted to 13 classical clinical and biochemical parameters. It does not integrate high-resolution modalities such as multi-lead raw electrocardiogram (ECG) waveforms, cardiac magnetic resonance imaging (MRI), CT coronary angiography DICOM scans, or polygenic risk scores.")
    add_body_paragraph(doc, "3. Cross-Sectional vs. Longitudinal Dynamics: The dataset captures static hospital admission indicators. Consequently, the study models immediate risk stratification rather than time-to-event survival analysis or long-term cardiovascular prognosis over sequential annual follow-ups.")
    add_body_paragraph(doc, "4. Pre-Clinical Software Scope: The developed decision support application functions as an investigational software prototype. It is not currently certified under regulatory frameworks (e.g., FDA SaMD or CE Mark) and requires multi-center prospective validation before deployment in active clinical workflows.")
    
    # Work Plan
    add_heading_1(doc, "• Work Plan (Week 1 to Week 8)")
    
    plan_table = doc.add_table(rows=9, cols=2)
    plan_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    plan_table.autofit = False
    
    col_w = [Inches(1.3), Inches(5.2)]
    headers = ["Week No.", "Activities Completed"]
    for c_idx, h in enumerate(headers):
        cell = plan_table.rows[0].cells[c_idx]
        cell.width = col_w[c_idx]
        set_cell_margins(cell, 100, 100, 120, 120)
        set_cell_background(cell, "002060") # Navy header
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    weeks_data = [
        ("Week 1", "a) Extensive literature review on machine learning in cardiovascular risk prediction.\nb) Identification of research gaps regarding probabilistic calibration and black-box triage.\nc) Formulation of project scope and selection of clinical evaluation metrics."),
        ("Week 2", "a) Ingestion and validation of the authentic UCI Cleveland dataset (N = 303).\nb) Exploratory data analysis (EDA), correlation profiling, and missing value checks.\nc) Drafting and formal submission of the Project Synopsis (Annexure 2)."),
        ("Week 3", "a) Implementation of data preprocessing pipeline and feature scaling (data_loader.py).\nb) Construction of leak-free Stratified 5-Fold Cross-Validation infrastructure.\nc) Definition of demographic subgroup slicing masks (Sex and Age cohorts)."),
        ("Week 4", "a) Implementation of baseline models across 21 paradigms (models.py).\nb) Execution of initial training rounds across linear, kernel, and tree architectures.\nc) Compilation and drafting of the Interim Report presentation (Annexure 3)."),
        ("Week 5", "a) Execution of automated hyperparameter tuning (RandomizedSearchCV) on top ensembles.\nb) Architecture design and training of the multi-model StackingClassifier meta-ensemble.\nc) Extraction and logging of optimal hyperparameter configurations."),
        ("Week 6", "a) Computation of 95% non-parametric bootstrap confidence intervals (500 resamples).\nb) Probabilistic calibration curve generation and Brier Score minimization.\nc) Execution of demographic fairness and disparity audits across patient slices."),
        ("Week 7", "a) Computation of SHAP global beeswarm and local waterfall force attributions.\nb) Development and responsive UI design of the interactive Streamlit dashboard (app.py).\nc) Recording and editing of the 5-minute video presentation (Annexure 4)."),
        ("Week 8", "a) Rigorous unit and integration test suite execution (test.py passing 100%).\nb) Writing and compiling the 45–65 page formal Project Report (Annexure 5).\nc) Running Turnitin/plagiarism checks (<= 20%) and final LMS portal submission.")
    ]
    
    for r_idx, (w, act) in enumerate(weeks_data, start=1):
        row = plan_table.rows[r_idx]
        cell_w = row.cells[0]
        cell_a = row.cells[1]
        
        cell_w.width = col_w[0]
        cell_a.width = col_w[1]
        set_cell_margins(cell_w, 80, 80, 100, 100)
        set_cell_margins(cell_a, 80, 80, 120, 120)
        
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        set_cell_background(cell_w, bg)
        set_cell_background(cell_a, bg)
        
        p_w = cell_w.paragraphs[0]
        p_w.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_w = p_w.add_run(w)
        r_w.font.name = 'Times New Roman'
        r_w.font.size = Pt(11)
        r_w.bold = True
        
        p_a = cell_a.paragraphs[0]
        p_a.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_a.paragraph_format.line_spacing = 1.2
        r_a = p_a.add_run(act)
        r_a.font.name = 'Times New Roman'
        r_a.font.size = Pt(10.5)

    docx_path = os.path.abspath("Benedict_Baah_Project_Synopsis.docx")
    doc.save(docx_path)
    print("Saved Synopsis docx:", docx_path)
    return docx_path

def build_interim_report():
    doc = create_base_document()
    add_header_banner(doc, "PROJECT INTERIM REPORT", "Research Methodology Document as per University Guidelines (Annexure 3)")
    add_meta_table(doc)
    
    # 1. Objectives of the Study
    add_heading_1(doc, "1. Objectives of the Study")
    add_body_paragraph(doc, "The primary objective of this empirical study is to eliminate diagnostic latency and overcome the 'black-box' barrier in clinical cardiovascular risk stratification through a calibrated, multi-algorithmic machine learning framework. The specific objectives are:")
    add_body_paragraph(doc, "• Systematic Algorithmic Benchmarking: To implement, train, and empirically benchmark 21 distinct machine learning paradigms spanning Generalized Linear Models (L1, L2, ElasticNet), Kernel Support Vector Machines, Bagged and Boosted Tree Ensembles (Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, AdaBoost), Artificial Neural Networks (MLP), Dimensionality Reduction (PCA, LDA), and Metric Classifiers (KNN).")
    add_body_paragraph(doc, "• Meta-Ensemble Formulation: To engineer a heterogeneous Stacking Classifier combining top-tier tree-based ensembles and regularized linear learners under an optimal meta-learner to maximize diagnostic discrimination.")
    add_body_paragraph(doc, "• Clinical Calibration & Probabilistic Reliability: To evaluate and minimize probabilistic error using Brier Score Loss and reliability diagrams, ensuring that model-generated percentages accurately reflect true clinical event rates rather than uncalibrated binary flags.")
    add_body_paragraph(doc, "• Demographic Subgroup Fairness Auditing: To perform slice-based fairness evaluations across patient biological sex (Male vs. Female) and chronological age (< 55 vs. >= 55 years) to detect, quantify, and mitigate algorithmic disparity.")
    add_body_paragraph(doc, "• Explainable AI (XAI) & Clinical Decision Support Deployment (Learner's Contribution): To integrate game-theoretic SHAP (SHapley Additive exPlanations) and deploy a fully interactive Streamlit Clinical Decision Support System (CDSS) delivering real-time patient risk quantification and visual biomarker attribution at the point of care.")
    
    # 2. Scope of the Study
    add_heading_1(doc, "2. Scope of the Study")
    add_body_paragraph(doc, "• Target Clinical Pathology: Risk stratification for Coronary Artery Disease (CAD), specifically angiographically documented coronary stenosis (>= 50% luminal diameter reduction in at least one major coronary vessel).")
    add_body_paragraph(doc, "• Clinical Setting & Cohort: Adult symptomatic patients undergoing diagnostic cardiac evaluation, represented by 13 standardized physiological, hemodynamic, and fluoroscopic biomarkers.")
    add_body_paragraph(doc, "• Methodological & Technical Scope:")
    add_body_paragraph(doc, "  - Comprehensive comparison of 21 supervised classification and unsupervised clustering algorithms.")
    add_body_paragraph(doc, "  - Rigorous hyperparameter tuning using randomized grid search across 6 leading model families.")
    add_body_paragraph(doc, "  - Statistical significance verification through 500-iteration non-parametric bootstrap resampling for 95% Confidence Intervals.")
    add_body_paragraph(doc, "  - Global feature ranking and personalized local risk attribution force plots via SHAP.")
    add_body_paragraph(doc, "  - Prototyping an interactive, browser-based decision support system for bedside triage.")
    add_body_paragraph(doc, "• Boundaries / Out of Scope: The study is an in silico retrospective investigation focusing on cross-sectional admission data. It does not replace definitive invasive catheterization and excludes unstructured high-dimensional modalities such as continuous ECG telemetry streams or DICOM angiographic imaging.")
    
    # 3. Methodology
    add_heading_1(doc, "3. Methodology")
    add_body_paragraph(doc, "The research methodology follows a closed-loop, data-leak-free engineering lifecycle across eight structured phases:")
    add_body_paragraph(doc, "1. Data Ingestion & Integrity Auditing: Ingestion of 303 patient records across 13 clinical biomarkers. Missing values are screened, data types (continuous, ordinal, binary) are validated, and clinical boundaries are audited.")
    add_body_paragraph(doc, "2. Leak-Free Preprocessing: To eliminate data leakage, feature standardization via Z-score transformation (mu = 0, sigma = 1) is fitted strictly on training partitions and applied to validation/test partitions.")
    add_body_paragraph(doc, "3. Stratified Cross-Validation & Tuning: 5-Fold Stratified Cross-Validation is established. Hyperparameters for Random Forest, XGBoost, LightGBM, CatBoost, SVM, and ElasticNet are tuned via RandomizedSearchCV optimizing for ROC-AUC.")
    add_body_paragraph(doc, "4. Stacking Architecture Formulation: Out-of-fold cross-validation probabilities from top-performing base learners are stacked and channeled into a regularized Logistic Regression meta-estimator to produce robust probability estimates.")
    add_body_paragraph(doc, "5. Multi-Dimensional Clinical Evaluation: Models are evaluated across Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC, and Matthews Correlation Coefficient (MCC).")
    add_body_paragraph(doc, "6. Probabilistic Calibration: Predicted probabilities are benchmarked against empirical outcomes using Brier Score loss and 10-bin calibration curves.")
    add_body_paragraph(doc, "7. Demographic Fairness Assessment: Holdout test cases are sliced into demographic subgroups (Male, Female, Age < 55, Age >= 55) to benchmark clinical sensitivity and false-negative rates.")
    add_body_paragraph(doc, "8. Explainability & Point-of-Care Deployment: SHAP calculates additive feature contributions. The entire analytical pipeline is operationalized into an interactive Streamlit application (app.py).")
    
    # 4. Research Design
    add_heading_1(doc, "4. Research Design")
    add_body_paragraph(doc, "• Research Paradigm: Quantitative, Empirical, and Comparative Experimental Research Design.")
    add_body_paragraph(doc, "• Experimental Controls: Fixed random state seed (random_state=42) enforced across all data splits, CV folds, and stochastic training procedures for exact reproducibility. Shared identical training (N = 242) and test (N = 61) partitions across all 21 algorithmic models to guarantee valid benchmark comparisons.")
    add_body_paragraph(doc, "• Variable Operationalization:")
    add_body_paragraph(doc, "  - Dependent Variable (Target): Binary cardiovascular status (0 = Absence [<50% stenosis], 1 = Presence [>= 50% stenosis]).")
    add_body_paragraph(doc, "  - Continuous Independent Features (5): age, resting blood pressure (trestbps), serum cholesterol (chol), maximum heart rate (thalach), ST depression (oldpeak).")
    add_body_paragraph(doc, "  - Categorical / Ordinal Features (5): chest pain type (cp: 0–3), resting ECG (restecg: 0–2), ST slope (slope: 0–2), fluoroscopy vessels (ca: 0–3), thallium scintigraphy (thal: 0–2).")
    add_body_paragraph(doc, "  - Binary Features (3): biological sex (sex), fasting blood sugar > 120 mg/dL (fbs), exercise-induced angina (exang).")
    
    # 5. Data Collection Method
    add_heading_1(doc, "5. Data Collection Method")
    add_body_paragraph(doc, "• Data Source: Secondary Data Collection using the canonical Cleveland Heart Disease Cohort, sourced from the UCI Machine Learning Repository (originating from the Cleveland Clinic Foundation, Ohio, USA, compiled by Dr. Robert Detrano).")
    add_body_paragraph(doc, "• Clinical Diagnostic Standard: Ground truth was established via coronary angiography (selective coronary arteriography), the clinical gold standard for CAD diagnosis.")
    add_body_paragraph(doc, "• Ethical & Regulatory Compliance: Secondary, de-identified clinical records compliant with standard data protection protocols (no Protected Health Information / PHI present).")
    add_body_paragraph(doc, "• Data Hygiene & Integrity: Dimensionality: 303 rows x 14 columns (13 clinical features + 1 target). Complete cases retained; boundary consistency verified (e.g., physiological limits on blood pressure and cholesterol).")
    
    # 6. Sampling Method (if applicable)
    add_heading_1(doc, "6. Sampling Method (if applicable)")
    add_body_paragraph(doc, "• Clinical Sampling: The original clinical dataset represents purposive/consecutive sampling of adult patients referred for coronary angiography following symptomatic cardiac presentations.")
    add_body_paragraph(doc, "• Partitioning & Resampling Strategy:")
    add_body_paragraph(doc, "  1. Stratified Holdout Split: An 80/20 train-test split (N_train = 242, N_test = 61) preserving the natural disease prevalence (54.46% disease absence vs. 45.54% disease presence) to evaluate out-of-sample generalization.")
    add_body_paragraph(doc, "  2. Stratified 5-Fold Cross-Validation: Applied across the training partition (k = 5), ensuring that each fold mirrors the class balance to prevent sampling bias during hyperparameter tuning.")
    add_body_paragraph(doc, "  3. Non-Parametric Bootstrap Resampling: A computational resampling procedure of B = 500 bootstrap iterations with replacement executed on test predictions to derive rigorous 95% Confidence Intervals for both ROC-AUC and F1-Scores.")
    
    # 7. Data Analysis Tools
    add_heading_1(doc, "7. Data Analysis Tools")
    add_body_paragraph(doc, "The computational environment and toolchain utilized for this research are outlined below:")
    
    tools_table = doc.add_table(rows=10, cols=4)
    tools_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tools_table.autofit = False
    
    t_widths = [Inches(1.5), Inches(1.5), Inches(1.0), Inches(2.5)]
    t_headers = ["Category", "Tool / Library", "Version", "Technical Role in Project"]
    for c_idx, h in enumerate(t_headers):
        cell = tools_table.rows[0].cells[c_idx]
        cell.width = t_widths[c_idx]
        set_cell_margins(cell, 100, 100, 100, 100)
        set_cell_background(cell, "002060")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    tools_data = [
        ("Runtime Environment", "Python", "3.11.16", "Base programming language in isolated virtual environment (.venv)"),
        ("Numerical Processing", "NumPy & Pandas", "1.26+", "Dataframe manipulation, feature transformation, and matrix arithmetic"),
        ("Machine Learning Core", "Scikit-Learn", "1.4+", "Model architectures, StandardScaler, RandomizedSearchCV, Stacking"),
        ("Gradient Boosters", "XGBoost\nLightGBM\nCatBoost", "2.0+\n4.3+\n1.2+", "Extreme Gradient Boosting, histogram-based LightGBM, and oblivious decision trees"),
        ("Statistical & Calibration", "Scikit-Learn Metrics", "1.4+", "brier_score_loss, matthews_corrcoef, calibration_curve, roc_auc_score"),
        ("Explainable AI (XAI)", "SHAP", "0.45+", "Cooperative game-theoretic Shapley value computations (TreeExplainer)"),
        ("Data Visualization", "Matplotlib & Seaborn", "3.8.4\n0.13+", "High-resolution publication plots (ROC, PR, calibration curves, beeswarm)"),
        ("Application & CDSS", "Streamlit", "1.32+", "Full-stack interactive clinical decision support web deployment (app.py)"),
        ("Version Control & CI", "Git & GitHub CLI", "2.55+", "Version tracking, CI automated testing (test.py), and remote repository management")
    ]
    
    for r_idx, (cat, tool, ver, role) in enumerate(tools_data, start=1):
        row = tools_table.rows[r_idx]
        row_data = [cat, tool, ver, role]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        
        for c_idx in range(4):
            cell = row.cells[c_idx]
            cell.width = t_widths[c_idx]
            set_cell_margins(cell, 80, 80, 100, 100)
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 2 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(row_data[c_idx])
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if c_idx == 0:
                r.bold = True
                
    docx_path = os.path.abspath("Benedict_Baah_Interim_Report.docx")
    doc.save(docx_path)
    print("Saved Interim docx:", docx_path)
    return docx_path

if __name__ == "__main__":
    s_docx = build_synopsis()
    i_docx = build_interim_report()
    print("Both DOCX generated successfully.")
