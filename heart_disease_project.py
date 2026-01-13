"""
HEART DISEASE PROJECT - FINAL VERSION
"""

print("=" * 50)
print("HEART DISEASE PROJECT SETUP")
print("=" * 50)

# Step 1: Setup
print("\nSTEP 1: Setting up...")
import subprocess
import sys
import os

# Create folders
for folder in ['data', 'models', 'results']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created {folder}/")

# Step 2: Install packages
print("\nSTEP 2: Checking packages...")
packages = ['pandas', 'numpy', 'scikit-learn', 'streamlit']

for package in packages:
    try:
        __import__(package)
        print(f"{package}: OK")
    except:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Step 3: Import
print("\nSTEP 3: Importing libraries...")
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Step 4: Create dataset
print("\nSTEP 4: Creating dataset...")
np.random.seed(42)
n = 100

data = {
    'age': np.random.randint(20, 80, n),
    'sex': np.random.randint(0, 2, n),
    'cp': np.random.randint(0, 4, n),
    'trestbps': np.random.randint(90, 200, n),
    'chol': np.random.randint(100, 400, n),
    'fbs': np.random.randint(0, 2, n),
    'restecg': np.random.randint(0, 3, n),
    'thalach': np.random.randint(60, 220, n),
    'exang': np.random.randint(0, 2, n),
    'oldpeak': np.round(np.random.uniform(0, 6, n), 1),
    'slope': np.random.randint(0, 3, n),
    'ca': np.random.randint(0, 4, n),
    'thal': np.random.randint(0, 4, n),
    'target': np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)
df.to_csv('data/heart.csv', index=False)
print(f"Dataset created: {len(df)} records")

# Step 5: Train models
print("\nSTEP 5: Training models...")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of models
models_list = [
    ('Logistic_Regression', LogisticRegression(max_iter=1000)),
    ('Random_Forest', RandomForestClassifier(n_estimators=100)),
    ('SVM', SVC(probability=True)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Decision_Tree', DecisionTreeClassifier()),
    ('Naive_Bayes', GaussianNB()),
    ('Logistic_L1', LogisticRegression(penalty='l1', solver='liblinear')),
    ('Logistic_L2', LogisticRegression(penalty='l2')),
    ('SVM_Linear', SVC(kernel='linear', probability=True)),
    ('SVM_RBF', SVC(kernel='rbf', probability=True)),
    ('KNN_3', KNeighborsClassifier(n_neighbors=3)),
    ('KNN_7', KNeighborsClassifier(n_neighbors=7)),
    ('Random_Forest_200', RandomForestClassifier(n_estimators=200)),
    ('Decision_Tree_Deep', DecisionTreeClassifier(max_depth=10)),
]

results = []
for name, model in models_list:
    try:
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)
        results.append({'Model': name, 'Accuracy': round(acc, 3)})
        
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"{name}: {acc:.3f}")
    except Exception as e:
        print(f"{name}: Error")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df.to_csv('results/model_results.csv', index=False)
print(f"\nTrained {len(results)} models")

# Step 6: Create app.py
print("\nSTEP 6: Creating app.py...")
app_code = 'import streamlit as st\n'
app_code += 'import pandas as pd\n'
app_code += 'import numpy as np\n'
app_code += 'import pickle\n\n'
app_code += 'st.set_page_config(page_title="Heart Predictor")\n\n'
app_code += 'st.title("Heart Disease Prediction")\n'
app_code += 'st.write("Enter patient details:")\n\n'
app_code += 'age = st.slider("Age", 20, 100, 50)\n'
app_code += 'sex = st.selectbox("Sex", ["Female", "Male"])\n'
app_code += 'bp = st.slider("Blood Pressure", 90, 200, 120)\n'
app_code += 'chol = st.slider("Cholesterol", 100, 600, 200)\n'
app_code += 'hr = st.slider("Heart Rate", 60, 220, 150)\n\n'
app_code += 'def load_model():\n'
app_code += '    try:\n'
app_code += '        with open("models/Random_Forest.pkl", "rb") as f:\n'
app_code += '            model = pickle.load(f)\n'
app_code += '        with open("models/scaler.pkl", "rb") as f:\n'
app_code += '            scaler = pickle.load(f)\n'
app_code += '        return model, scaler\n'
app_code += '    except:\n'
app_code += '        return None, None\n\n'
app_code += 'if st.button("Predict"):\n'
app_code += '    model, scaler = load_model()\n'
app_code += '    if model and scaler:\n'
app_code += '        features = np.array([[age, 1 if sex=="Male" else 0, 0, bp, chol, 0, 0, hr, 0, 1.0, 1, 0, 2]])\n'
app_code += '        features_scaled = scaler.transform(features)\n'
app_code += '        pred = model.predict(features_scaled)[0]\n'
app_code += '        prob = model.predict_proba(features_scaled)[0][1]\n'
app_code += '        if pred == 1:\n'
app_code += '            st.error(f"High Risk: {prob:.1%}")\n'
app_code += '        else:\n'
app_code += '            st.success(f"Low Risk: {prob:.1%}")\n'
app_code += '        st.progress(float(prob))\n'
app_code += '    else:\n'
app_code += '        st.error("Model not loaded")\n\n'
app_code += 'st.write("---")\n'
app_code += 'st.write("Final Project Submission.")\n'

with open('app.py', 'w') as f:
    f.write(app_code)
print("Created app.py")

# Step 7: Create test.py
print("\nSTEP 7: Creating test.py...")
test_code = 'print("Testing Heart Project")\n'
test_code += 'print("=" * 40)\n'
test_code += 'import os\n'
test_code += 'print("Folders:")\n'
test_code += 'for f in ["data", "models", "results"]:\n'
test_code += '    if os.path.exists(f): print(f"  {f}/ OK")\n'
test_code += '    else: print(f"  {f}/ MISSING")\n'
test_code += 'print("\\nFiles:")\n'
test_code += 'for f in ["data/heart.csv", "models/Random_Forest.pkl"]:\n'
test_code += '    if os.path.exists(f): print(f"  {f} OK")\n'
test_code += '    else: print(f"  {f} MISSING")\n'
test_code += 'print("\\n" + "=" * 40)\n'
test_code += 'print("TEST COMPLETE")\n'
test_code += 'print("\\nRun: streamlit run app.py")\n'
test_code += 'print("Then open: http://localhost:8501")\n'

with open('test.py', 'w') as f:
    f.write(test_code)
print("Created test.py")

# Step 8: Create requirements.txt
print("\nSTEP 8: Creating requirements.txt...")
with open('requirements.txt', 'w') as f:
    f.write('pandas\nnumpy\nscikit-learn\nstreamlit\n')
print("Created requirements.txt")

# Step 9: Create README
print("\nSTEP 9: Creating README.md...")
readme_text = '# Heart Disease Project\n\n'
readme_text += '## How to run:\n'
readme_text += '1. pip install -r requirements.txt\n'
readme_text += '2. python test.py\n'
readme_text += '3. streamlit run app.py\n'
readme_text += '4. Open http://localhost:8501\n\n'
readme_text += '## Note:\n'
readme_text += 'Educational purpose only.\n'

with open('README.md', 'w') as f:
    f.write(readme_text)
print("Created README.md")

# Done
print("\n" + "=" * 50)
print("PROJECT CREATED SUCCESSFULLY!")
print("=" * 50)
print("\nFiles created:")
print("  data/heart.csv")
print("  models/*.pkl (14 models)")
print("  results/model_results.csv")
print("  app.py")
print("  test.py")
print("  requirements.txt")
print("  README.md")
print("\nTo run:")
print("  1. python test.py")
print("  2. streamlit run app.py")
print("\nGood luck!")