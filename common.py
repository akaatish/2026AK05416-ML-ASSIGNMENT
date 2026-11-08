"""Shared helpers: NSL-KDD loading and preprocessing."""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL]


def load_nsl_kdd(path):
    """Load a raw NSL-KDD file (tab- or comma-separated, with or without header).

    Returns a DataFrame with the 41 feature columns plus `label` (original
    attack name) and `target` (0 = normal, 1 = attack).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    has_header = first.strip().lower().startswith("duration")
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None)
    if has_header:
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={"class": "label", "attack": "label"})
    else:
        extra = ["label", "difficulty"][: df.shape[1] - len(COLUMNS)]
        df.columns = COLUMNS + extra
    df = df.drop(columns=[c for c in ["difficulty", "difficulty_level"] if c in df.columns])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["target"] = (df["label"] != "normal").astype(int)
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
