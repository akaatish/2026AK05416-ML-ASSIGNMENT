"""Streamlit app: NSL-KDD network intrusion detection with five classifiers."""
import os

import pandas as pd
import streamlit as st

import utils

st.set_page_config(page_title="NSL-KDD Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_artifacts():
    return utils.load_artifacts()


@st.cache_data
def get_default_data():
    return pd.read_csv(os.path.join(utils.BASE_DIR, "test_data.csv"))


@st.cache_data
def get_metrics_table():
    return pd.read_csv(os.path.join(utils.MODEL_DIR, "metrics.csv"))


@st.cache_data(show_spinner="Running model...")
def run_model(model_name, df):
    pre, models = get_artifacts()
    X_raw, y = utils.prepare(df)
    pred, score = utils.predict(models[model_name], pre.transform(X_raw))
    return pred, score, y


st.title("🛡️ Network Intrusion Detection (NSL-KDD)")
st.write(
    "Classifies network connections as **Normal** or **Attack** using five machine learning "
    "models trained on the NSL-KDD dataset (`KDDTrain+`) and evaluated on `KDDTest+`."
)

try:
    pre, models = get_artifacts()
except Exception as e:  # e.g. scikit-learn version mismatch with the saved models
    st.error(
        "Could not load the saved models. Re-create them with "
        "`python train_models.py --train KDDTrain+.txt --test KDDTest+.txt`.\n\n" + str(e)
    )
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select model", list(utils.MODEL_FILES.keys()), index=1)
source = st.sidebar.radio("Data source", ["Bundled test_data.csv (KDDTest+)", "Upload CSV"])

df = None
if source.startswith("Bundled"):
    df = get_default_data()
else:
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    st.sidebar.caption(
        "Needs the 41 NSL-KDD feature columns (same header as test_data.csv). "
        "An optional `label` column (normal / attack) enables evaluation metrics."
    )
    if up is not None:
        df = pd.read_csv(up)

tab_cmp, tab_eval = st.tabs(["📊 Model comparison", "🔍 Evaluate / predict"])

# ---------------- Tab 1: comparison ----------------
with tab_cmp:
    st.subheader("Evaluation metrics on KDDTest+ (22,544 records)")
    table = get_metrics_table()
    st.dataframe(table.set_index("ML Model Name").style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 use_container_width=True)
    st.bar_chart(table.set_index("ML Model Name")[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]])
    st.caption("KNN and SVM were trained on a stratified 20,000-row subsample; the other models on all 125,973 rows.")

# ---------------- Tab 2: evaluate / predict ----------------
with tab_eval:
    if df is None:
        st.info("Upload a CSV in the sidebar to get predictions.")
    else:
        try:
            pred, score, y = run_model(model_name, df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.subheader(f"{model_name}: {len(df):,} records")
        n_attack = int(pred.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Predicted attacks", f"{n_attack:,}")
        c3.metric("Predicted normal", f"{len(df) - n_attack:,}")

        if y is not None and len(set(y)) == 2:
            m = utils.compute_metrics(y, pred, score)
            cols = st.columns(6)
            for col, (k, v) in zip(cols, m.items()):
                col.metric(k, f"{v:.4f}")
            g1, g2 = st.columns(2)
            g1.pyplot(utils.confusion_matrix_fig(y, pred, f"Confusion matrix: {model_name}"))
            g2.pyplot(utils.roc_fig(y, score, f"ROC curve: {model_name}"))
        elif y is not None:
            st.warning("The label column contains only one class, so AUC and related metrics are not shown.")
        else:
            st.info("No `label` column found, so only predictions are shown (no evaluation metrics).")

        out = pd.DataFrame({
            "prediction": ["Attack" if p == 1 else "Normal" for p in pred],
            "attack_score": score.round(4),
        })
        if y is not None:
            out.insert(0, "actual", ["Attack" if v == 1 else "Normal" for v in y])
        st.subheader("Predictions (first 200 rows)")
        st.dataframe(out.head(200), use_container_width=True)
        st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
