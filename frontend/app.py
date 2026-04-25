import os
import random

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def sample_transaction() -> dict:
    payload = {
        "Time": float(random.randint(0, 172792)),
        "Amount": round(random.uniform(1, 2500), 2),
    }
    for index in range(1, 29):
        payload[f"V{index}"] = round(random.uniform(-3, 3), 4)
    return payload


st.set_page_config(page_title="Fraud Detection", page_icon=":credit_card:", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Containerized real-time inference demo")

with st.sidebar:
    st.subheader("Service")
    st.write(API_URL)
    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=3)
            response.raise_for_status()
            st.success("API is healthy")
        except requests.RequestException as exc:
            st.error(f"Health check failed: {exc}")

defaults = sample_transaction()

left, right = st.columns([1, 1])
with left:
    amount = st.number_input("Amount", min_value=0.0, value=defaults["Amount"], step=10.0)
    transaction_time = st.number_input(
        "Time", min_value=0.0, value=defaults["Time"], step=100.0
    )

features = {}
with right:
    st.write("PCA Features")
    cols = st.columns(4)
    for index in range(1, 29):
        with cols[(index - 1) % 4]:
            features[f"V{index}"] = st.number_input(
                f"V{index}", value=defaults[f"V{index}"], format="%.4f"
            )

payload = {"Time": transaction_time, "Amount": amount, **features}

if st.button("Score Transaction", type="primary", use_container_width=True):
    with st.spinner("Scoring transaction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            status = "Fraud flagged" if result["fraud"] else "Likely legitimate"
            metric_cols = st.columns(3)
            metric_cols[0].metric("Decision", status)
            metric_cols[1].metric("Fraud probability", f"{result['confidence']:.4f}")
            metric_cols[2].metric("Transaction ID", result["transaction_id"][:8])
            st.json(result)

st.divider()

try:
    alerts_response = requests.get(f"{API_URL}/alerts", timeout=3)
    alerts_response.raise_for_status()
    alerts = alerts_response.json()
except requests.RequestException:
    alerts = []

st.subheader("Recent Fraud Alerts")
if alerts:
    st.dataframe(alerts, use_container_width=True, hide_index=True)
else:
    st.info("No fraud alerts recorded in this API container yet.")
