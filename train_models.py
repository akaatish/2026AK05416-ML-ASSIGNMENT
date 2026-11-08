"""
Train the five classifiers on NSL-KDD (binary: normal vs attack), evaluate on
KDDTest+, and save models + artifacts into ./model.

Usage:
    python train_models.py --train KDDTrain+.txt --test KDDTest+.txt
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import COLUMNS, build_preprocessor, load_nsl_kdd

RANDOM_STATE = 42
SLOW_TRAIN_SIZE = 20000  # KNN / SVM use a stratified subsample of this size


def scores_for(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="KDDTrain+.txt")
    ap.add_argument("--test", default="KDDTest+.txt")
    ap.add_argument("--model_dir", default="model")
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    train, test = load_nsl_kdd(args.train), load_nsl_kdd(args.test)
    X_train, y_train = train[COLUMNS], train["target"]
    X_test, y_test = test[COLUMNS], test["target"]

    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    Xtr_s, _, ytr_s, _ = train_test_split(
        Xtr, y_train, train_size=SLOW_TRAIN_SIZE, stratify=y_train, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), False),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        "Naive Bayes": (GaussianNB(), False),
        "Support Vector Machine": (SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE), True),
    }
    rows = []
    for name, (m, slow) in models.items():
        Xf, yf = (Xtr_s, ytr_s) if slow else (Xtr, y_train)
        t0 = time.time()
        m.fit(Xf, yf)
        pred, sc = m.predict(Xte), scores_for(m, Xte)
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, sc),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(m, os.path.join(args.model_dir, fname), compress=3)
        print(f"{name:25s} done in {time.time()-t0:6.1f}s  trained on {len(yf)} rows")

    joblib.dump(pre, os.path.join(args.model_dir, "preprocessor.joblib"), compress=3)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(args.model_dir, "metrics.csv"), index=False)
    json.dump({"features": COLUMNS, "knn_svm_train_size": SLOW_TRAIN_SIZE},
              open(os.path.join(args.model_dir, "feature_columns.json"), "w"), indent=2)
    test.drop(columns=["target"]).to_csv("test_data.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
