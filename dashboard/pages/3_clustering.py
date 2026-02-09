import streamlit as st
import sys

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 247, 237, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(249, 115, 22, 0.15);
        border: 1px solid rgba(254, 215, 170, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #f97316 0%, #fb923c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
    }
    .stNumberInput input {
        border-radius: 8px;
        border: 2px solid rgba(254, 215, 170, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add Models folder using absolute path
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

from clustering_model import predict_cluster, get_cluster_profile

st.title("🧬 Patient Segmentation")
st.write("Find out which patient group this person belongs to")

st.subheader("📋 Patient Information")

# Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age Code (0-9)", 0, 9, 5)
    time_in_hospital = st.number_input("Days in Hospital", 1, 30, 4)
    num_lab_procedures = st.number_input("Lab Tests", 0, 200, 40)
    num_procedures = st.number_input("Procedures Done", 0, 50, 1)
    num_medications = st.number_input("Medications", 0, 200, 15)

with col2:
    number_outpatient = st.number_input("Past Outpatient Visits", 0, 50, 0)
    number_emergency = st.number_input("Past ER Visits", 0, 50, 0)
    number_inpatient = st.number_input("Past Hospitalizations", 0, 50, 0)
    number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 8)

# Button to predict
if st.button("🔍 Find Patient Group"):
    
    # Put inputs in dictionary
    patient_data = {
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses
    }
    
    # Get prediction
    cluster = predict_cluster(patient_data)
    profile = get_cluster_profile(cluster)
    
    st.write("---")
    
    # Show result with colors
    if cluster == 0:
        st.warning(f"🟡 **Cluster {cluster}: Older, High Hospital Use**")
    elif cluster == 1:
        st.success(f"🟢 **Cluster {cluster}: Low-Risk Patient**")
    elif cluster == 2:
        st.error(f"🔴 **Cluster {cluster}: High-Complexity Patient**")
    elif cluster == 3:
        st.info(f"🔵 **Cluster {cluster}: Chronic-Care Patient**")
    else:
        st.warning(f"🟠 **Cluster {cluster}: Very High Utilization**")
    
    st.write(f"**Description:** {profile}")
    
    # Check for contradictions
    contradictions = []
    
    if cluster == 4 and age >= 6:
        contradictions.append("⚠️ WARNING: Cluster 4 is typically for younger patients (40-50), but this patient is elderly")
    
    if cluster == 1 and (num_medications >= 20 or time_in_hospital >= 7):
        contradictions.append("⚠️ WARNING: Cluster 1 is for low-risk patients, but this patient has high complexity indicators")
    
    if cluster == 2 and age <= 4:
        contradictions.append("⚠️ WARNING: Cluster 2 is typically for elderly patients, but this patient is young")
    
    if contradictions:
        for warning in contradictions:
            st.warning(warning)
        st.info("💡 This patient has mixed characteristics. Consider reviewing manually.")
    
    # Show more details
    with st.expander("📊 More Details About This Group"):
        
        # Cluster 0
        if cluster == 0:
            st.write("**Who are they?**")
            st.write("- Older patients (60-70)")
            st.write("- High past inpatient visits (2+)")
            st.write("- Moderate medications and labs")
            st.write("- Moderate hospital stays")
            
            st.write("")
            st.write("**What to do:**")
            st.write("⚠️ Monitor for readmissions")
            st.write("⚠️ Chronic disease management")
            st.write("⚠️ Regular follow-ups")
        
        # Cluster 1
        elif cluster == 1:
            st.write("**Who are they?**")
            st.write("- Middle-aged, healthy patients")
            st.write("- Lowest medications and lab tests")
            st.write("- Very low past hospital visits")
            st.write("- Shortest hospital stays")
            
            st.write("")
            st.write("**What to do:**")
            st.write("✅ Normal discharge planning")
            st.write("✅ Basic follow-up")
            st.write("✅ Patient education")
        
        # Cluster 2
        elif cluster == 2:
            st.write("**Who are they?**")
            st.write("- Oldest patients (70+)")
            st.write("- Longest hospital stays (8+ days)")
            st.write("- Highest medications and lab tests")
            st.write("- Highest procedures")
            st.write("- Very complex cases")
            
            st.write("")
            st.write("**What to do:**")
            st.write("🚨 Intensive care needed")
            st.write("🚨 Multiple specialists")
            st.write("🚨 Home care or nursing facility")
            st.write("🚨 Close follow-up within 7 days")
        
        # Cluster 3
        elif cluster == 3:
            st.write("**Who are they?**")
            st.write("- Older patients (60-70)")
            st.write("- High medications and lab tests")
            st.write("- Very LOW past hospital visits")
            st.write("- Chronic conditions but stable")
            
            st.write("")
            st.write("**What to do:**")
            st.write("💊 Focus on medication management")
            st.write("💊 Regular lab monitoring")
            st.write("💊 Strong primary care connection")
        
        # Cluster 4
        else:
            st.write("**Who are they?**")
            st.write("- Youngest patients (40-50)")
            st.write("- **VERY HIGH past inpatient visits (5+)**")
            st.write("- **VERY HIGH ER visits (4+)**")
            st.write("- **VERY HIGH outpatient visits (3+)**")
            st.write("- Frequent hospital users")
            
            st.write("")
            st.write("**What to do:**")
            st.write("🔄 Find why they keep coming back")
            st.write("🔄 Social worker help")
            st.write("🔄 Mental health screening")
            st.write("🔄 Care management program")
        
        st.write("---")
        st.write("**Patient Summary:**")
        
        # Simple risk check
        if num_medications >= 20:
            st.write("🔴 High medications - Complex case")
        elif num_medications <= 10:
            st.write("🟢 Low medications - Simple case")
        
        if number_inpatient >= 2:
            st.write("🔴 Past hospitalizations - Chronic issues")
        elif number_inpatient == 0:
            st.write("🟢 First time in hospital")
        
        if number_emergency >= 3:
            st.write("🔴 Multiple ER visits")
        
        if age >= 7:
            st.write("🔴 Elderly - Needs more care")
        elif age <= 3:
            st.write("🟢 Young - Recovers faster")
        
        if time_in_hospital >= 8:
            st.write("🔴 Very long stay - Complex case")
        elif time_in_hospital <= 3:
            st.write("🟢 Short stay - Quick recovery")
