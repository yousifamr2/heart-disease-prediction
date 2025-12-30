from preprocessing import find_outliers
from preprocessing import handdling_outliers
from visual import visuailation_outliers
from Analytics import load_data
from model import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


if __name__ == "__main__":
    train_df = load_data("data/heart_statlog_cleveland_hungary_final.csv")
    train_df = handdling_outliers(train_df)

    model = Model(
        target_col="target",   
        random_state=42,
        test_size=0.2
    )


    X_train, X_test, y_train, y_test = model.prepare_data(train_df)


    model.training_model(X_train, y_train, X_test, y_test)


    stacking_clf = model.stacking_model(
        train_df.drop(columns=["target"]),
        train_df["target"]
    )


    voting_clf = model.voting_model(
        train_df.drop(columns=["target"]),
        train_df["target"]
    )



    # stats = find_outliers(train_df, "resting bp s")
    # print(stats)
    
    # visuailation_outliers(train_df, "resting bp s")

    # for col in train_df.columns:
    #     stats = find_outliers(train_df, col)
    #     print(stats)
    #     visuailation_outliers(train_df, col)