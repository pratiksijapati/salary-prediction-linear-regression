import os, joblib, numpy as np, json
import streamlit as st
import subprocess, sys

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "model.joblib")
SCALER_PATH = os.path.join(HERE, "scaler.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")
TRAIN_SCRIPT = os.path.join(HERE, "train.py")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Supervised Learning — Salary Prediction")
st.caption("Linear Regression on Years of Experience")

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return True
    if st.button("Train model now"):
        result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
        st.code(result.stdout + "\n" + result.stderr or "")
        st.experimental_rerun()
    st.warning("Model not found. Click the button above to train it.")
    return False

if ensure_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    years = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
    if st.button("Predict Salary"):
        X = np.array([[years]])
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        st.success(f"Estimated Salary: **₹ {pred:,.0f}**")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        st.subheader("Model Metrics (test split)")
        st.json({k: round(v, 4) for k, v in m.items()})
