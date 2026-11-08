"""Helper functions used by the Streamlit app (kept separate so they can be tested without Streamlit)."""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURES = json.load(f)["features"]


def load_artifacts(model_dir=MODEL_DIR):
    """Return (preprocessor, {model name: fitted model})."""
    pre = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    models = {n: joblib.load(os.path.join(model_dir, fn)) for n, fn in MODEL_FILES.items()}
    return pre, models


def prepare(df):
    """Validate columns and split into raw features and (optional) binary target.

    The target is read from a `label` column (anything other than "normal" is an
    attack) or from a 0/1 `target` column. Returns (X_raw, y or None).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    y = None
    if "label" in df.columns:
        y = (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int).to_numpy()
    elif "target" in df.columns:
        y = df["target"].astype(int).to_numpy()
    return df[FEATURES], y


def predict(model, X_encoded):
    """Return (predicted class, continuous attack score)."""
    pred = model.predict(X_encoded)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_encoded)[:, 1]
    else:
        score = model.decision_function(X_encoded)
    return pred, score


def compute_metrics(y, pred, score):
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def confusion_matrix_fig(y, pred, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(confusion_matrix(y, pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def roc_fig(y, score, title):
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, score):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
