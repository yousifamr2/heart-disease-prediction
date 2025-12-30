import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
                                accuracy_score, 
                                f1_score, 
                                classification_report, 
                                ConfusionMatrixDisplay, 
                                confusion_matrix, 
                                recall_score, 
                                precision_score
                                )
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
                                RandomForestClassifier, 
                                AdaBoostClassifier, 
                                GradientBoostingClassifier, 
                                VotingClassifier, 
                                StackingClassifier)
from sklearn.svm import SVC


try:
    from xgboost import XGBClassifier  
except ImportError:  
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier 
except ImportError:  
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier 
except ImportError: 
    CatBoostClassifier = None

Random_State = 42
Test_Size = 0.2
Target_Col = "target"

class Model:
    def __init__(self, target_col = Target_Col, random_state = Random_State, test_size = Test_Size):
        self.best_models = {}
        self.param_grids = {}
        self.random_state = random_state
        self.test_size = test_size
        self.target_col = target_col
        
        self.base_models= {
            "LogisticRegression": LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1, random_state=self.random_state),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
        }

        # Add optional models only if their libraries are installed
        if XGBClassifier is not None:
            self.base_models["XGBoost"] = XGBClassifier(
                random_state=self.random_state,
                verbosity=0,
                eval_metric="logloss",
            )

        if LGBMClassifier is not None:
            self.base_models["LightGBM"] = LGBMClassifier(random_state=self.random_state, verbosity=-1)

        if CatBoostClassifier is not None:
            self.base_models["CatBoost"] = CatBoostClassifier(random_seed=self.random_state, verbose=0)

        self.param_grids = {
            "LogisticRegression": {
                "C": [0.1, 1.0, 10.0], 
                "max_iter": [500]
                },
            "RandomForest": {
                "n_estimators": [100, 200], 
                "max_depth": [None, 5, 10]
                },
            "AdaBoost": {
                "n_estimators": [100, 200], 
                "learning_rate": [0.05, 0.1, 0.5]
                },
            "GradientBoosting": {
                "n_estimators": [100, 200], 
                "learning_rate": [0.05, 0.1], 
                "max_depth": [1, 3, 5]
                },
        }

        if "XGBoost" in self.base_models:
            self.param_grids["XGBoost"] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }

        if "LightGBM" in self.base_models:
            self.param_grids["LightGBM"] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [-1, 5],
            }

        if "CatBoost" in self.base_models:
            self.param_grids["CatBoost"] = {
                "iterations": [100, 200],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6],
            }
    def prepare_data(self, data):
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]

        return train_test_split(
                                X,
                                y, 
                                test_size=self.test_size, 
                                random_state=self.random_state, 
                                stratify=y
                                )
    
    def training_model(self, X_train, y_train, X_test, y_test):
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            if name in self.param_grids:
                grid = GridSearchCV(
                    model, 
                    self.param_grids[name], 
                    cv=StratifiedKFold(
                        n_splits=3,
                        shuffle=True, 
                        random_state=self.random_state
                        ), 
                    scoring='accuracy', 
                    n_jobs=-1
                    )
                try:
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    print(f"Best params for {name}: {grid.best_params_}")
                except AttributeError as e:
                    # Common with some 3rd-party estimators + newer scikit-learn:
                    # e.g. CatBoostClassifier missing __sklearn_tags__.
                    print(f"[WARN] GridSearch failed for {name} ({e}). Falling back to plain fit().")
                    model.fit(X_train, y_train)
                    best_model = model
            else:
                model.fit(X_train, y_train)
                best_model = model
            self.best_models[name] = best_model
            self.evaluate_model(best_model, X_test, y_test, name)

    def evaluate_model(self, model, X_test, y_test, name):
        val_preds = model.predict(X_test)
        acc = accuracy_score(y_test, val_preds)
        recall = recall_score(y_test, val_preds, average="weighted")
        precision = precision_score(y_test, val_preds, average="weighted")
        f1 = f1_score(y_test, val_preds, average="weighted")
        cm = confusion_matrix(y_test, val_preds)
        class_report = classification_report(y_test, val_preds)

        print(
                f"{name} =>\n"
                f"Accuracy : {acc:.4f}\n"
                f"Recall   : {recall:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"F1 Score : {f1:.4f}\n"
                f"Confusion Matrix:\n{cm}\n"
                f"Classification Report:\n{class_report}"
            )

        
    def stacking_model(self, X, y):

        estimators = []
        for name, mdl in self.best_models.items():
            if hasattr(mdl, "__sklearn_tags__"):
                estimators.append((name, mdl))
            else:
                print(f"[WARN] Excluding {name} from stacking (missing __sklearn_tags__).")

        if not estimators:
            raise ValueError("No compatible estimators available for stacking.")

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            n_jobs=-1
        )
        stacking_clf.fit(X, y)
        return stacking_clf
    
    def voting_model(self, X, y):
        estimators = []
        for name, mdl in self.best_models.items():
            if hasattr(mdl, "__sklearn_tags__"):
                estimators.append((name, mdl))
            else:
                print(f"[WARN] Excluding {name} from voting (missing __sklearn_tags__).")

        if not estimators:
            raise ValueError("No compatible estimators available for voting.")

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        voting_clf.fit(X, y)
        return voting_clf
