import streamlit as st
import sys
import os
import joblib
import numpy as np

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e6f0fa 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(100, 116, 139, 0.15);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model directly here
model_path = r"D:\HealthCare System\Trained_Models\Readmission\xgb_readmission_final_model.pkl"
scaler_path = r"D:\HealthCare System\Trained_Models\Readmission\scaler_readmission_final.pkl"
features_path = r"D:\HealthCare System\Trained_Models\Readmission\selected_features_final.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_list = joblib.load(features_path)

def predict_readmission(patient_data):
    """Predict readmission"""
    print("\n" + "="*60)
    print("🔍 DEBUG: Feature Matching")
    print("="*60)
    print(f"Model expects {len(feature_list)} features:")
    
    values = []
    for i, feature in enumerate(feature_list, 1):
        value = patient_data.get(feature, 0)
        values.append(value)
        print(f"{i}. {feature}: {value}")
    
    print(f"\n📊 Values sent to model: {values}")
    
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = 1 if probability >= 0.3 else 0
    
    print(f"🎯 Probability: {probability:.3f}")
    print(f"✅ Prediction: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
    print("="*60 + "\n")
    
    return prediction, probability

st.title("🏥 30-Day Readmission Predictor")

st.write("Fill in patient details to predict readmission risk")

# Simple input form
number_inpatient = st.number_input("Previous Hospitalizations", 0, 50, 0, help="Number of times admitted as inpatient in past year")
number_outpatient = st.number_input("Outpatient Visits (past year)", 0, 100, 0, help="Number of outpatient clinic visits")
number_emergency = st.number_input("Emergency Visits", 0, 50, 0, help="Number of ER visits in past year")
num_medications = st.number_input("Number of Medications", 0, 200, 15, help="Total medications prescribed")
time_in_hospital = st.number_input("Days in Hospital", 1, 30, 5, help="Current hospitalization length")
num_lab_procedures = st.number_input("Lab Tests Done", 0, 500, 50, help="Number of lab procedures performed")

# Technical fields (hidden - use defaults)
st.markdown("---")
with st.expander("⚙️ Advanced/Technical Fields (Optional)"):
    medical_specialty = st.number_input("Medical Specialty Code", 0, 100, 0, help="Leave 0 if unknown")
    diag_3 = st.number_input("Third Diagnosis Code", 0, 999, 0, help="ICD code for 3rd diagnosis (leave 0 if none)")
    discharge_desc = st.number_input("Discharge Description Code", 0, 50, 0, help="Leave 0 for default")
    admission_source_desc = st.number_input("Admission Source Description Code", 0, 50, 0, help="Leave 0 for default")

# User-friendly dropdowns
discharge_options = {
    "Home": 1,
    "Transfer to Another Facility": 2,
    "Home with Health Service": 3,
    "Expired": 11,
    "Other": 5
}
discharge_choice = st.selectbox("Discharge Type", list(discharge_options.keys()))
discharge_disposition_id = discharge_options[discharge_choice]

admission_type_options = {
    "Emergency": 1,
    "Urgent": 2,
    "Elective": 3,
    "Newborn": 4,
    "Trauma": 5
}
admission_type_choice = st.selectbox("Admission Type", list(admission_type_options.keys()))
admission_type_id = admission_type_options[admission_type_choice]

admission_source_options = {
    "Physician Referral": 1,
    "Clinic Referral": 2,
    "Emergency Room": 7,
    "Transfer from Hospital": 4,
    "Normal Delivery": 17
}
admission_source_choice = st.selectbox("Admission Source", list(admission_source_options.keys()))
admission_source_id = admission_source_options[admission_source_choice]

age_options = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9
}
age_choice = st.selectbox("Age Range", list(age_options.keys()), index=5)
age = age_options[age_choice]

# Primary diagnosis with common options
diag_options = {
    "Diabetes (250)": 250,
    "Heart Failure (428)": 428,
    "Coronary Artery Disease (414)": 414,
    "Pneumonia (486)": 486,
    "COPD (496)": 496,
    "Kidney Disease (585)": 585,
    "Other/Custom": 0
}
diag_choice = st.selectbox("Primary Diagnosis", list(diag_options.keys()), help="Main medical condition")
if diag_choice == "Other/Custom":
    diag_1 = st.number_input("Enter ICD Code", 1, 999, 250)
else:
    diag_1 = diag_options[diag_choice]

st.markdown("---")

# Predict button
if st.button("🔮 Predict Readmission Risk"):
    
    patient_data = {
        'discharge_disposition_id': discharge_disposition_id,
        'time_in_hospital': time_in_hospital,
        'medical_specialty': medical_specialty,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'diag_1': diag_1,
        'diag_3': diag_3,
        'discharge_desc': discharge_desc,
        'admission_source_desc': admission_source_desc
    }
    
    # Show input values for debugging
    with st.expander("🔍 Debug: Input Values"):
        st.json(patient_data)
    
    prediction, probability = predict_readmission(patient_data)
    
    st.write("---")
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK")
        st.write(f"**Readmission chance: {probability:.1%}**")
        st.info("💡 Recommendation: Patient needs follow-up care")
    else:
        st.success(f"✅ LOW RISK")
        st.write(f"Readmission chance: {probability:.1%}")
        st.info("Patient can be safely discharged")
