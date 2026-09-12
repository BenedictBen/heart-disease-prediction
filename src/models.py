"""
Model Definitions Module: 21+ Supervised, Ensembles, Neural Nets, Stacking, and Unsupervised Algorithms
Includes Automated Hyperparameter Tuning via RandomizedSearchCV and Stacking Meta-Ensemble.
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
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def get_all_models(include_stacking=True):
    """
    Returns a dictionary of algorithms configured with baseline and tuned hyperparameters.
    Categorized into Supervised Classifiers, Dimensionality Reduction + Classifiers,
    Stacking Meta-Ensemble, and Unsupervised Clustering algorithms.
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
        "CatBoost": cb.CatBoostClassifier(iterations=120, depth=4, learning_rate=0.08, verbose=0, random_seed=42, allow_writing_files=False),
        
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
    
    if include_stacking:
        supervised_models["Stacking_Classifier"] = build_stacking_classifier()
    
    unsupervised_models = {
        # 6. Unsupervised Clustering
        "KMeans_Clustering": KMeans(n_clusters=2, n_init=10, random_state=42),
        "Hierarchical_Clustering": AgglomerativeClustering(n_clusters=2, linkage='ward'),
        "DBSCAN_Clustering": DBSCAN(eps=2.5, min_samples=4)
    }
    
    return supervised_models, unsupervised_models


def build_stacking_classifier(base_estimators=None):
    """
    Constructs a meta-learning Stacking Classifier combining complementary model families
    (Tree ensembles, Gradient Boosting, Regularized Linear Models) with a Logistic Regression meta-learner.
    """
    if base_estimators is None:
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=3, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)),
            ('cat', cb.CatBoostClassifier(iterations=100, depth=3, learning_rate=0.05, verbose=0, random_seed=42, allow_writing_files=False)),
            ('lr', LogisticRegression(C=0.8, penalty='l2', max_iter=1000, random_state=42))
        ]
    
    final_meta_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_meta_estimator,
        cv=5,
        n_jobs=-1
    )
    return stacking_clf


def get_tuning_param_distributions():
    """
    Returns search spaces for key model architectures for hyperparameter optimization.
    """
    return {
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_split": [2, 3, 5, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['sqrt', 'log2', None]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [80, 120, 160, 220],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.06, 0.1, 0.15],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.3, 0.5]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(verbose=-1, random_state=42),
            "params": {
                "n_estimators": [80, 120, 180, 240],
                "max_depth": [3, 4, 5, 7],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.04, 0.08, 0.12],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
                "reg_lambda": [0.0, 0.1, 0.5, 1.0]
            }
        },
        "CatBoost": {
            "model": cb.CatBoostClassifier(verbose=0, random_seed=42, allow_writing_files=False),
            "params": {
                "iterations": [80, 120, 180, 250],
                "depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.05, 0.08, 0.12],
                "l2_leaf_reg": [1, 3, 5, 7]
            }
        },
        "Logistic_Regression_ElasticNet": {
            "model": LogisticRegression(penalty='elasticnet', solver='saga', max_iter=2500, random_state=42),
            "params": {
                "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        "SVM_RBF": {
            "model": SVC(kernel='rbf', probability=True, random_state=42),
            "params": {
                "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "gamma": ['scale', 'auto', 0.01, 0.05, 0.1, 0.2]
            }
        }
    }


def tune_top_models(X_train, y_train, n_iter=10):
    """
    Performs hyperparameter search across key architectures using Stratified 5-Fold Cross Validation.
    Returns a dictionary of best tuned estimators and their optimal parameters.
    """
    param_configs = get_tuning_param_distributions()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_models = {}
    tuning_summary = []
    
    print("\n" + "-" * 60)
    print("HYPERPARAMETER OPTIMIZATION (Stratified 5-Fold CV)")
    print("-" * 60)
    
    for name, config in param_configs.items():
        print(f"[*] Tuning {name} ({n_iter} iterations)...")
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=1 if name == "CatBoost" else -1,
            error_score='raise'
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_score = search.best_score_
        tuned_models[name] = best_model
        
        tuning_summary.append({
            "Model": name,
            "Best CV ROC-AUC": round(best_score, 4),
            "Best Parameters": search.best_params_
        })
        print(f"    -> Best CV ROC-AUC: {best_score:.4f} with {search.best_params_}")
        
    print("-" * 60)
    return tuned_models, tuning_summary

