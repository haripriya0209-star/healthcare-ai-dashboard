import streamlit as st
import pandas as pd
import numpy as np
import sys

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe8e8 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(254, 242, 242, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(254, 226, 226, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .stNumberInput input {
        border-radius: 8px;
        border: 2px solid rgba(254, 226, 226, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add Models folder using absolute path
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

from lstm_model import predict_risk

st.title("📈 LSTM ICU Risk Prediction")

FEATURES = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Derived_HRV",
    "Derived_Pulse_Pressure",
    "Derived_MAP"
]

st.write("Choose one option below to provide patient vitals.")

# ---------------------------------------------------------
# OPTION A: CSV UPLOAD
# ---------------------------------------------------------
st.subheader("Option A: Upload CSV")

uploaded = st.file_uploader("Upload a CSV with vitals", type=["csv"])
sequence = None

if uploaded:
    df = pd.read_csv(uploaded)

    if list(df.columns) != FEATURES:
        st.error("Column names do not match the required format.")
        st.write("Expected columns:", FEATURES)
    else:
        st.success("CSV loaded successfully.")
        st.dataframe(df)
        sequence = df.values


# ---------------------------------------------------------
# OPTION B: MANUAL ENTRY (Easy Mode!)
# ---------------------------------------------------------
st.subheader("Option B: Enter Vitals Manually")

st.info("💡 Enter vital signs for 3 time points (measurements taken at different times)")

# Initialize session state for scenario
if 'scenario' not in st.session_state:
    st.session_state.scenario = "healthy"

# Quick test buttons
st.write("**Quick Test Scenarios:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💚 Healthy Patient"):
        st.session_state.scenario = "healthy"
with col2:
    if st.button("🟢 Normal Patient"):
        st.session_state.scenario = "normal"
with col3:
    if st.button("🟡 Borderline Patient"):
        st.session_state.scenario = "borderline"
with col4:
    if st.button("🔴 Critical Patient"):
        st.session_state.scenario = "critical"

# Set values based on scenario
if st.session_state.scenario == "healthy":
    vals = [(60, 14, 36.8, 99, 110, 70, 0.18, 40, 83.33),
            (62, 15, 36.9, 99, 112, 72, 0.17, 40, 85.33),
            (58, 13, 36.7, 99, 108, 68, 0.19, 40, 81.33)]
elif st.session_state.scenario == "normal":
    vals = [(75, 18, 37.1, 97, 122, 81, 0.14, 41, 94.67),
            (76, 18, 37.2, 97, 123, 82, 0.14, 41, 95.67),
            (74, 17, 37.0, 97, 121, 80, 0.15, 41, 93.67)]
elif st.session_state.scenario == "borderline":
    vals = [(88, 21, 37.9, 93, 142, 90, 0.09, 52, 107.33),
            (89, 22, 38.0, 93, 143, 91, 0.09, 52, 108.33),
            (87, 20, 37.8, 93, 141, 89, 0.10, 52, 106.33)]
else:  # critical
    vals = [(130, 32, 39.0, 85, 170, 100, 0.04, 70, 123.33),
            (132, 33, 39.1, 85, 172, 102, 0.04, 70, 125.33),
            (128, 31, 38.9, 86, 168, 98, 0.05, 70, 121.33)]

# This list will store all the data
all_data = []

# TIME STEP 1 - First measurement
st.write("### ⏰ Time Step 1")
hr1 = st.number_input("Heart Rate", value=float(vals[0][0]), key="hr_1")
rr1 = st.number_input("Respiratory Rate", value=float(vals[0][1]), key="rr_1")
temp1 = st.number_input("Body Temperature", value=float(vals[0][2]), key="temp_1")
o2_1 = st.number_input("Oxygen Saturation", value=float(vals[0][3]), key="o2_1")
sbp1 = st.number_input("Systolic Blood Pressure", value=float(vals[0][4]), key="sbp_1")
dbp1 = st.number_input("Diastolic Blood Pressure", value=float(vals[0][5]), key="dbp_1")
hrv1 = st.number_input("HRV (Heart Rate Variability)", value=float(vals[0][6]), key="hrv_1")
pp1 = st.number_input("Pulse Pressure", value=float(vals[0][7]), key="pp_1")
map1 = st.number_input("MAP (Mean Arterial Pressure)", value=float(vals[0][8]), key="map_1")

# Save first row of data
row1 = [hr1, rr1, temp1, o2_1, sbp1, dbp1, hrv1, pp1, map1]
all_data.append(row1)

st.divider()

# TIME STEP 2 - Second measurement
st.write("### ⏰ Time Step 2")
hr2 = st.number_input("Heart Rate", value=float(vals[1][0]), key="hr_2")
rr2 = st.number_input("Respiratory Rate", value=float(vals[1][1]), key="rr_2")
temp2 = st.number_input("Body Temperature", value=float(vals[1][2]), key="temp_2")
o2_2 = st.number_input("Oxygen Saturation", value=float(vals[1][3]), key="o2_2")
sbp2 = st.number_input("Systolic Blood Pressure", value=float(vals[1][4]), key="sbp_2")
dbp2 = st.number_input("Diastolic Blood Pressure", value=float(vals[1][5]), key="dbp_2")
hrv2 = st.number_input("HRV (Heart Rate Variability)", value=float(vals[1][6]), key="hrv_2")
pp2 = st.number_input("Pulse Pressure", value=float(vals[1][7]), key="pp_2")
map2 = st.number_input("MAP (Mean Arterial Pressure)", value=float(vals[1][8]), key="map_2")

# Save second row of data
row2 = [hr2, rr2, temp2, o2_2, sbp2, dbp2, hrv2, pp2, map2]
all_data.append(row2)

st.divider()

# TIME STEP 3 - Third measurement
st.write("### ⏰ Time Step 3")
hr3 = st.number_input("Heart Rate", value=float(vals[2][0]), key="hr_3")
rr3 = st.number_input("Respiratory Rate", value=float(vals[2][1]), key="rr_3")
temp3 = st.number_input("Body Temperature", value=float(vals[2][2]), key="temp_3")
o2_3 = st.number_input("Oxygen Saturation", value=float(vals[2][3]), key="o2_3")
sbp3 = st.number_input("Systolic Blood Pressure", value=float(vals[2][4]), key="sbp_3")
dbp3 = st.number_input("Diastolic Blood Pressure", value=float(vals[2][5]), key="dbp_3")
hrv3 = st.number_input("HRV (Heart Rate Variability)", value=float(vals[2][6]), key="hrv_3")
pp3 = st.number_input("Pulse Pressure", value=float(vals[2][7]), key="pp_3")
map3 = st.number_input("MAP (Mean Arterial Pressure)", value=float(vals[2][8]), key="map_3")

# Save third row of data
row3 = [hr3, rr3, temp3, o2_3, sbp3, dbp3, hrv3, pp3, map3]
all_data.append(row3)

st.divider()

# Button to predict
if st.button("🔍 Predict Risk from Manual Entry", type="primary"):
    # Convert our list into numpy array (what the model needs)
    sequence = np.array(all_data)
    
    # Show what was entered in a nice table
    with st.expander("📊 View Entered Data"):
        df_display = pd.DataFrame(sequence, columns=FEATURES)
        df_display.index = ["Time Step 1", "Time Step 2", "Time Step 3"]
        st.dataframe(df_display)
    
    # Make prediction (will be shown in the section below)
    sequence = np.array(all_data)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if sequence is not None:
    st.subheader("Prediction Result")
    label, prob = predict_risk(sequence)

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {prob:.3f}")
    
    # Warning for extreme confidence scores
    if prob > 0.95 or prob < 0.05:
        st.warning("⚠️ **Note:** Model showing very high confidence. This may indicate overfitting. In production, confidence scores should be recalibrated using techniques like Platt scaling or temperature scaling.")

    if label == "High Risk":
        st.error("⚠️ High deterioration risk detected.")
    else:
        st.success("🟢 Low deterioration risk detected.")
