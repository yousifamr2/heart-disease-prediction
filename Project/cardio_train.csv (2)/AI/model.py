import pandas as pd
import numpy as np
import joblib
import os

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
        """
        
        Prepare data for training and testing.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to prepare
        Returns:
        --------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Testing features
        y_train : pd.Series
            Training target
        y_test : pd.Series
        Returns:
        """
        
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

        """
        Train the models.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
        """
        
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

        """
        Evaluate the model.
        
        Parameters:
        -----------
        model : model
            Model to evaluate
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
        name : str
            Name of the model
        """
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

        """
        Stack the models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        """
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
        """
        Vote the models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        Returns:
        --------
        voting_clf : VotingClassifier
            Voting classifier
        """
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
    
    def save_best_model(self, X_test, y_test, save_path="models/best_model.pkl", metric="accuracy"):
        """
        Save the best model based on performance metric.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features for evaluation
        y_test : pd.Series
            Test target for evaluation
        save_path : str
            Path to save the model (default: "models/best_model.pkl")
        metric : str
            Metric to use for selecting best model (default: "accuracy")
            Options: "accuracy", "precision", "recall", "f1_score"
        
        Returns:
        --------
        str : Name of the best model
        dict : Metrics of the best model
        """
        if not self.best_models:
            raise ValueError("No models trained yet. Call training_model() first.")
        
        # Evaluate all models and find the best one
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model_scores = {}
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            
            if metric == "accuracy":
                score = accuracy_score(y_test, y_pred)
            elif metric == "precision":
                score = precision_score(y_test, y_pred, average="weighted")
            elif metric == "recall":
                score = recall_score(y_test, y_pred, average="weighted")
            elif metric == "f1_score":
                score = f1_score(y_test, y_pred, average="weighted")
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            model_scores[name] = score
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        best_model = self.best_models[best_model_name]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Save the model
        joblib.dump(best_model, save_path)
        
        # Get all metrics for the best model
        y_pred = best_model.predict(X_test)
        best_metrics = {
            "model_name": best_model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        
        print(f"\n{'='*60}")
        print(f"Best Model Saved: {best_model_name}")
        print(f"{'='*60}")
        print(f"Metric used: {metric}")
        print(f"Score: {model_scores[best_model_name]:.4f}")
        print(f"\nAll Metrics:")
        print(f"  Accuracy : {best_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall   : {best_metrics['recall']:.4f}")
        print(f"  F1 Score : {best_metrics['f1_score']:.4f}")
        print(f"\nModel saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return best_model_name, best_metrics
    
    @staticmethod
    def load_model(model_path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        
        Returns:
        --------
        model : Trained model object
        """
        return joblib.load(model_path)
