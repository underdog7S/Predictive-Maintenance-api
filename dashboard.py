import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests

# Page Config
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance - AI Dashboard")
st.markdown("Use the sidebar to input machine parameters and predict potential failures.")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Machine Parameters")

    product_type = st.selectbox("Product Type", ['L', 'M', 'H'])
    air_temp = st.slider("Air Temperature (°C)", 10.0, 100.0, 35.0)
    process_temp = st.slider("Process Temperature (°C)", 20.0, 120.0, 60.0)
    rot_speed = st.slider("Rotational Speed (rpm)", 500.0, 2000.0, 1500.0)
    torque = st.slider("Torque (Nm)", 0.0, 300.0, 60.0)
    tool_wear = st.slider("Tool Wear (min)", 0.0, 300.0, 50.0)

    st.markdown("### Failure Type Indicators")
    TWF = st.number_input("Tool Wear Failure (TWF)", value=0)
    HDF = st.number_input("Heat Dissipation Failure (HDF)", value=0)
    PWF = st.number_input("Power Failure (PWF)", value=0)
    OSF = st.number_input("Overstrain Failure (OSF)", value=0)
    RNF = st.number_input("Random Failure (RNF)", value=0)

# Correct encoding to match API: L=0, M=1, H=2
type_dict = {'L': 0, 'M': 1, 'H': 2}
type_encoded = type_dict[product_type]

# Prepare input dictionary exactly as API expects
api_input = {
    "product_type": product_type,
    "air_temperature": air_temp,
    "process_temperature": process_temp,
    "rotational_speed": rot_speed,
    "torque": torque,
    "tool_wear": tool_wear,
    "twf": int(TWF),
    "hdf": int(HDF),
    "pwf": int(PWF),
    "osf": int(OSF),
    "rnf": int(RNF),
}

# Create DataFrame for displaying inputs and logs (optional)
input_df = pd.DataFrame([{
    'Type': type_encoded,
    'Air temperature [K]': air_temp + 273.15,
    'Process temperature [K]': process_temp + 273.15,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'TWF': TWF,
    'HDF': HDF,
    'PWF': PWF,
    'OSF': OSF,
    'RNF': RNF
}])

# Prediction button and logic
if st.button("🔮 Run Prediction"):

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=api_input)
        response.raise_for_status()
        prediction_data = response.json()

        prediction = prediction_data["prediction"]
        normal_prob = round(prediction_data.get("normal_probability", 0) * 100, 2)
        failure_prob = round(prediction_data.get("failure_probability", 0) * 100, 2)
    except Exception as e:
        st.error(f"❌ Failed to get prediction from API: {e}")
        prediction = None

    if prediction is not None:
        # Results Summary
        st.subheader("🔍 Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("🟢 No Failure Probability", f"{normal_prob}%", delta=f"{normal_prob - 50:.2f}")
        col2.metric("🔴 Failure Probability", f"{failure_prob}%", delta=f"{failure_prob - 50:.2f}")

        if prediction == 1:
            st.error(f"⚠️ Machine Failure Likely (Confidence: {failure_prob}%)")
        else:
            st.success(f"✅ Machine Operating Normally (Confidence: {normal_prob}%)")

        # Input Summary Chart
        with st.expander("📈 Input Feature Summary"):
            numeric_cols = input_df.select_dtypes(include=np.number).columns
            fig, ax = plt.subplots()
            ax.barh(numeric_cols, input_df[numeric_cols].iloc[0], color='skyblue')
            ax.set_xlabel("Value")
            ax.set_title("Input Feature Overview")
            st.pyplot(fig)

        # Suggestions Section
        with st.expander("💡 Recommendations"):
            suggestions = []

            if torque > 120:
                suggestions.append("🔧 High Torque: Check for shaft misalignment or overload.")
            if tool_wear > 150:
                suggestions.append("🛠️ Tool wear is high. Consider tool replacement.")
            if OSF > 50:
                suggestions.append("📦 Overstrain risk. Inspect load and reduce stress.")
            if air_temp > 40:
                suggestions.append("🌡️ Elevated air temp. Enhance ventilation.")
            if process_temp > 75:
                suggestions.append("🔥 High process temp. Verify sensors.")
            if rot_speed > 1700:
                suggestions.append("⚙️ High speed. Check motor balance.")
            if HDF > 0:
                suggestions.append("💨 Heat issue. Clean cooling components.")
            if PWF > 0:
                suggestions.append("🔌 Power instability. Inspect connections.")
            if RNF > 0:
                suggestions.append("❗ Random issue detected. Run full diagnostics.")

            if suggestions:
                for s in suggestions:
                    st.warning(s)
            else:
                st.success("✅ All values within optimal range.")

        # Risk Flags
        with st.expander("🚨 Risk Flags"):
            risk_flags = []

            if torque > 120:
                risk_flags.append("⚠️ High Torque")
            if tool_wear > 150:
                risk_flags.append("⚠️ Excessive Tool Wear")
            if OSF > 50:
                risk_flags.append("⚠️ Overstrain Risk")
            if air_temp > 40:
                risk_flags.append("⚠️ High Air Temp")
            if process_temp > 75:
                risk_flags.append("⚠️ High Process Temp")
            if rot_speed > 1700:
                risk_flags.append("⚠️ High Speed")
            if HDF > 0:
                risk_flags.append("⚠️ Heat Issue")
            if PWF > 0:
                risk_flags.append("⚠️ Power Failure")
            if RNF > 0:
                risk_flags.append("⚠️ Random Failure")

            if risk_flags:
                for flag in risk_flags:
                    st.warning(flag)
            else:
                st.success("✅ No risk flags triggered.")

        # Logging
        log_data = input_df.copy()
        log_data["Prediction"] = "Failure" if prediction == 1 else "No Failure"
        log_data["Failure_Prob"] = failure_prob
        log_data["Normal_Prob"] = normal_prob
        log_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data["Risk_Flags"] = "; ".join(risk_flags) if risk_flags else "None"

        log_cols = [
            'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
            'Prediction', 'Failure_Prob', 'Normal_Prob', 'Timestamp', 'Risk_Flags'
        ]
        log_data = log_data[log_cols]

        log_file = "prediction_log.csv"
        log_data.to_csv(log_file, mode='a' if os.path.exists(log_file) else 'w', header=not os.path.exists(log_file), index=False)

        # Download Button
        with open(log_file, "rb") as f:
            st.download_button("📥 Download Log", f, file_name="prediction_log.csv")

else:
    st.info("Set machine parameters and click 🔮 **Run Prediction** to see results.")
