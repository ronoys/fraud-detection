import os
import time

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = {
    "Legitimate example 1": {
        "amount": 0.76, "time": 29501.0,
        "V1": 1.1789, "V2": 0.4944, "V3": -0.3844, "V4": 0.7178,
        "V5": 0.1849, "V6": -0.7375, "V7": 0.2082, "V8": -0.0351,
        "V9": -0.4103, "V10": -0.4676, "V11": 2.185, "V12": 0.672,
        "V13": -0.3634, "V14": -0.7438, "V15": 0.4922, "V16": 0.3107,
        "V17": 0.9266, "V18": 0.1394, "V19": -0.4682, "V20": -0.1156,
        "V21": -0.0473, "V22": -0.0915, "V23": -0.0348, "V24": 0.1326,
        "V25": 0.4354, "V26": 0.3689, "V27": -0.0222, "V28": 0.0248,
    },
    "Legitimate example 2": {
        "amount": 2.99, "time": 66183.0,
        "V1": 0.9242, "V2": 0.7073, "V3": -0.4924, "V4": 1.0256,
        "V5": 0.3586, "V6": -0.5193, "V7": 0.2625, "V8": -1.0457,
        "V9": -0.1301, "V10": 0.2146, "V11": -1.3418, "V12": -0.6034,
        "V13": -0.9009, "V14": 0.8337, "V15": 0.965, "V16": -0.1301,
        "V17": -0.3651, "V18": 0.1347, "V19": 0.4822, "V20": -0.297,
        "V21": 0.7793, "V22": -0.1689, "V23": -0.0798, "V24": -0.4658,
        "V25": 0.3584, "V26": -0.2586, "V27": 0.2137, "V28": 0.2335,
    },
    "Legitimate example 3": {
        "amount": 168.33, "time": 87695.0,
        "V1": -2.1423, "V2": -2.4956, "V3": 0.0172, "V4": 0.3618,
        "V5": 3.112, "V6": -2.7159, "V7": 0.1672, "V8": -0.3993,
        "V9": 0.1013, "V10": -0.3915, "V11": -1.0525, "V12": -0.0669,
        "V13": -0.2561, "V14": 0.5124, "V15": 0.3721, "V16": -0.9022,
        "V17": -0.3416, "V18": 0.1205, "V19": 0.458, "V20": 1.0153,
        "V21": 0.4909, "V22": 0.6094, "V23": 0.5969, "V24": -0.1264,
        "V25": 0.0059, "V26": -0.5686, "V27": -0.1215, "V28": 0.0371,
    },
    "Fraudulent example 1": {
        "amount": 1.0, "time": 96717.0,
        "V1": -3.7059, "V2": 4.1079, "V3": -3.8037, "V4": 1.7103,
        "V5": -3.5825, "V6": 1.4697, "V7": -9.6216, "V8": -11.9131,
        "V9": -0.3223, "V10": -6.6257, "V11": 2.1752, "V12": -4.3811,
        "V13": 2.0633, "V14": -0.6738, "V15": 1.4008, "V16": -4.2549,
        "V17": -5.1602, "V18": -1.3025, "V19": 2.5945, "V20": 3.6396,
        "V21": -5.4988, "V22": 2.9415, "V23": 0.9162, "V24": -0.2555,
        "V25": -0.1838, "V26": -0.5845, "V27": -0.3155, "V28": -0.0972,
    },
    "Fraudulent example 2": {
        "amount": 1.0, "time": 91502.0,
        "V1": 0.0074, "V2": 2.3652, "V3": -2.6003, "V4": 1.1116,
        "V5": 3.2764, "V6": -1.7761, "V7": 2.1145, "V8": -0.8301,
        "V9": 0.9005, "V10": -3.3762, "V11": 2.0568, "V12": -3.9843,
        "V13": 1.022, "V14": -5.9679, "V15": -1.1516, "V16": 1.6797,
        "V17": 5.5861, "V18": 2.7891, "V19": -2.2411, "V20": -0.0064,
        "V21": -0.5639, "V22": -0.9021, "V23": -0.4044, "V24": -0.0129,
        "V25": 0.5898, "V26": -0.7344, "V27": -0.4475, "V28": -0.3624,
    },
}


def risk_label(confidence: float) -> str:
    if confidence < 0.3:
        return "Low"
    if confidence < 0.6:
        return "Medium"
    return "High"


st.set_page_config(page_title="Fraud Detection", page_icon=":credit_card:", layout="centered")

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

example_choice = st.selectbox("Transaction profile", list(EXAMPLES.keys()))
ex = EXAMPLES[example_choice]

col1, col2 = st.columns(2)
col1.metric("Amount ($)", f"{ex['amount']:.2f}")
col2.metric("Time (s)", f"{ex['time']:.0f}")

payload = {
    "Time": ex["time"],
    "Amount": ex["amount"],
    **{k: v for k, v in ex.items() if k.startswith("V")},
}

if st.button("Score Transaction", type="primary", use_container_width=True):
    result = None
    error = None

    with st.status("Running fraud detection pipeline...", expanded=True) as status:
        st.write("**Step 1** — Transaction received")
        time.sleep(0.4)
        st.write("**Step 2** — Normalizing amount with StandardScaler")
        time.sleep(0.4)
        st.write("**Step 3** — Running XGBoost inference")
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            error = str(exc)
        time.sleep(0.3)
        st.write("**Step 4** — Applying decision threshold")
        time.sleep(0.3)
        if error:
            status.update(label="Pipeline failed", state="error")
        else:
            status.update(label="Pipeline complete", state="complete", expanded=False)

    if error:
        st.error(f"Prediction failed: {error}")
    elif result:
        st.subheader("Pipeline trace")
        for step in result.get("pipeline_steps", []):
            st.markdown(f"- **{step['name']}** — {step['detail']}")

        st.subheader("Decision")
        is_fraud = result["fraud"]
        metric_cols = st.columns(3)
        metric_cols[0].metric("Classification", "Fraud" if is_fraud else "Legitimate")
        metric_cols[1].metric("Risk level", risk_label(result["confidence"]))
        metric_cols[2].metric("Transaction ID", result["transaction_id"][:8])

        if st.button("Send SMS Alert", use_container_width=True):
            try:
                notify_response = requests.post(
                    f"{API_URL}/notify",
                    json={
                        "amount": ex["amount"],
                        "timestamp": result["timestamp"],
                        "transaction_id": result["transaction_id"],
                    },
                    timeout=10,
                )
                notify_response.raise_for_status()
                nr = notify_response.json()
                st.success(f"SMS sent: \"{nr['message']}\"")
            except requests.RequestException as exc:
                st.error(f"SMS failed: {exc}")

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
