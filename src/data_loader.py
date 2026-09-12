"""
Data Loader & Preprocessing Module for UCI Cleveland Heart Disease Dataset
"""

import os
import shutil
import pandas as pd
import numpy as np

DATA_DIR = "data"
LOCAL_CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_NAME = "target"

# Clinical mapping for human-readable interpretability
CLINICAL_MAPPINGS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    },
    "fbs": {0: "< 120 mg/dl (Normal)", 1: "> 120 mg/dl (Elevated)"},
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown / Reversible"}
}

def load_and_prepare_data():
    """
    Loads authentic UCI Cleveland dataset, handles missing values/formatting,
    and saves a canonical clean version to data/heart.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Priority 1: Check canonical local CSV
    if os.path.exists(LOCAL_CSV_PATH):
        df = pd.read_csv(LOCAL_CSV_PATH)
    else:
        # Fallback to UCI direct URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(url, names=columns, na_values="?")
    
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure target is binary (0 = healthy, 1 = heart disease)
    # In raw UCI data, target ranges 0-4; >0 indicates presence of heart disease
    if 'target' in df.columns:
        df['target'] = (df['target'] > 0).astype(int)
    elif 'condition' in df.columns:
        df['target'] = df['condition'].astype(int)
        df.drop(columns=['condition'], inplace=True)
    elif 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
        df.drop(columns=['num'], inplace=True)

    # Impute or drop any null values if present
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Save canonical local copy
    df.to_csv(LOCAL_CSV_PATH, index=False)
    print(f"[DataLoader] Clean dataset saved to {LOCAL_CSV_PATH} with shape: {df.shape}")
    print(f"[DataLoader] Class balance: {dict(df['target'].value_counts())}")
    
    return df

if __name__ == "__main__":
    load_and_prepare_data()
