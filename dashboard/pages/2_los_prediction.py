import streamlit as st
import sys

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 253, 244, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(220, 252, 231, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #22c55e 0%, #4ade80 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }
    .stNumberInput input {
        border-radius: 8px;
        border: 2px solid rgba(220, 252, 231, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add Models folder using absolute path
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

from LOS_model import predict_los

st.title("🏥 Length of Stay Predictor")
st.write("Predict how many days a patient will stay in the hospital")

# Input fields
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    num_medications = st.number_input("Number of Medications", 0, 200, 15)
    num_lab_procedures = st.number_input("Lab Procedures", 0, 500, 50)
    number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 8)
    num_procedures = st.number_input("Number of Procedures", 0, 50, 1)
    discharge_disposition_id = st.number_input("Discharge Disposition ID", 1, 30, 1)

with col2:
    age = st.number_input("Age Code (0-9)", 0, 9, 5)
    number_inpatient = st.number_input("Previous Hospitalizations", 0, 50, 0)
    number_emergency = st.number_input("Emergency Visits", 0, 50, 0)
    
    # Secondary diagnosis dropdown with common conditions
    diag_2_options = {
        "None/Other": 0,
        "Diabetes": 250,
        "Hypertension ": 401,
        "Heart Failure": 428,
        "Kidney Disease": 585
    }
    diag_2_selected = st.selectbox("Secondary Diagnosis", list(diag_2_options.keys()), index=1)
    diag_2 = diag_2_options[diag_2_selected]
    
    A1Cresult = st.selectbox("A1C Result", [0, 1, 2, 3], index=3, 
                             format_func=lambda x: ["None", "Normal", ">7%", ">8%"][x])
    insulin = st.selectbox("Insulin Usage", [0, 1, 2, 3], index=0,
                          format_func=lambda x: ["No", "Down", "Up", "Steady"][x])

# Predict button
if st.button("🔮 Predict Length of Stay"):
    
    patient_data = {
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_diagnoses': number_diagnoses,
        'num_procedures': num_procedures,
        'discharge_disposition_id': discharge_disposition_id,
        'age': age,
        'number_inpatient': number_inpatient,
        'diag_2': diag_2,
        'A1Cresult': A1Cresult,
        'insulin': insulin,
        'number_emergency': number_emergency
    }
    
    los_days = predict_los(patient_data)
    
    st.write("---")
    
    # Color-code based on length
    if los_days < 3:
        st.success(f"✅ **Short Stay: {los_days:.1f} days**")
        st.info("Patient expected to be discharged soon")
    elif los_days < 7:
        st.warning(f"⚠️ **Moderate Stay: {los_days:.1f} days**")
        st.info("Standard hospitalization period")
    else:
        st.error(f"🏥 **Extended Stay: {los_days:.1f} days**")
        st.info("Long-term care planning needed")
    
    # Show breakdown
    with st.expander("📊 Prediction Details"):
        st.write(f"**Predicted Days:** {los_days:.2f}")
        st.write(f"**Model:** XGBoost Regressor (Tuned)")
        
        st.markdown("---")
        st.subheader("📈 Feature Impact Analysis")
        
        # Show which features are HIGH vs LOW
        st.write("**High Impact Factors (increasing stay):**")
        impacts = []
        
        if num_medications > 20:
            impacts.append(f"🔴 High Medications ({num_medications}) - Complex treatment")
        if num_lab_procedures > 60:
            impacts.append(f"🔴 Many Lab Tests ({num_lab_procedures}) - Intensive monitoring")
        if number_diagnoses > 10:
            impacts.append(f"🔴 Multiple Diagnoses ({number_diagnoses}) - Complicated case")
        if num_procedures > 3:
            impacts.append(f"🔴 Multiple Procedures ({num_procedures}) - Major interventions")
        if number_inpatient > 2:
            impacts.append(f"🔴 Prior Hospitalizations ({number_inpatient}) - Chronic conditions")
        if age >= 7:
            impacts.append(f"🔴 Elderly Patient (Age {['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'][age]}) - Longer recovery")
        
        # Show diagnosis impact
        diag_names = {250: "Diabetes", 401: "Hypertension", 428: "Heart Failure", 585: "Kidney Disease"}
        if diag_2 > 0:
            diag_name = diag_names.get(diag_2, f"Code {diag_2}")
            impacts.append(f"🟡 Secondary Diagnosis: {diag_name} - Chronic condition present")
        
        if impacts:
            for impact in impacts:
                st.write(impact)
        else:
            st.write("🟢 No high-risk factors identified")
        
        st.write("")
        st.write("**Low Impact Factors (shorter stay):**")
        positives = []
        
        if num_medications <= 10:
            positives.append(f"🟢 Low Medications ({num_medications}) - Less complex")
        if num_lab_procedures <= 30:
            positives.append(f"🟢 Few Lab Tests ({num_lab_procedures}) - Stable condition")
        if number_diagnoses <= 5:
            positives.append(f"🟢 Few Diagnoses ({number_diagnoses}) - Simple case")
        if number_inpatient == 0:
            positives.append(f"🟢 No Prior Hospitalizations - First admission")
        if age <= 4:
            positives.append(f"🟢 Younger Patient (Age {['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'][age]}) - Faster recovery")
        
        if positives:
            for positive in positives:
                st.write(positive)
        else:
            st.write("⚠️ Multiple risk factors present")

