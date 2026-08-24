"""
Model Definitions Module: 21 Supervised, Ensembles, Neural Nets, and Unsupervised Algorithms
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_all_models():
    """
    Returns a dictionary of all 21 algorithms configured with appropriate hyperparameters.
    Categorized into Supervised Classifiers, Dimensionality Reduction + Classifiers,
    and Unsupervised Clustering algorithms.
    """
    supervised_models = {
        # 1. Generalized Linear Models
        "Logistic_Regression_L2": LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
        "Logistic_Regression_L1": LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000, random_state=42),
        "Logistic_Regression_ElasticNet": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=2000, random_state=42),
        
        # 2. Instance-based & Probabilistic
        "Gaussian_Naive_Bayes": GaussianNB(),
        "KNN_Classifier": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        "SVM_Linear": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        "SVM_RBF": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        
        # 3. Tree-based & Ensembles
        "Decision_Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=3, random_state=42),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, verbose=-1, random_state=42),
        "CatBoost": cb.CatBoostClassifier(iterations=120, depth=4, learning_rate=0.08, verbose=0, random_seed=42),
        
        # 4. Neural Networks (Deep Learning)
        "Artificial_Neural_Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            alpha=0.001,
            early_stopping=True,
            random_state=42
        ),
        
        # 5. Dimensionality Reduction + Supervised
        "PCA_Logistic_Regression": Pipeline([
            ('pca', PCA(n_components=6, random_state=42)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Linear_Discriminant_Analysis": LinearDiscriminantAnalysis(solver='svd')
    }
    
    unsupervised_models = {
        # 6. Unsupervised Clustering
        "KMeans_Clustering": KMeans(n_clusters=2, n_init=10, random_state=42),
        "Hierarchical_Clustering": AgglomerativeClustering(n_clusters=2, linkage='ward'),
        "DBSCAN_Clustering": DBSCAN(eps=2.5, min_samples=4)
    }
    
    return supervised_models, unsupervised_models
