import streamlit as st
import pandas as pd
import sys

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 50%, #ffffff 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #eab308 0%, #fbbf24 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.3);
    }
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid rgba(254, 240, 138, 0.5);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Add Models folder using absolute path
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

from association_model import get_top_rules, search_rules

st.title("📌 Association Rule Mining")

st.write("This page shows patterns found in patient data.")
st.write("These patterns show which things happen together.")

# Simple explanation
with st.expander("ℹ️ Click Here to Understand This Page"):
    st.write("**What Are Association Rules?**")
    st.write("They show patterns like:")
    st.write("IF → Male patient + Few medications + Readmitted after 30 days")
    st.write("THEN → Short hospital stay")
    
    st.write("")
    st.write("**What Do The Numbers Mean?**")
    st.write("- **Support**: How often this pattern happens")
    st.write("- **Confidence**: How reliable (0.76 = happens 76% of time)")
    st.write("- **Lift**: If > 1, it's a real pattern (not random)")
    
    st.write("")
    st.write("**What Can You Search For?**")
    st.write("")
    st.write("**Hospital Stay Length:**")
    st.write("- Short = 0-3 days")
    st.write("- Medium = 4-6 days")  
    st.write("- Long = 7+ days")
    
    st.write("")
    st.write("**Medication Amount:**")
    st.write("- Low = Few medications (0-10)")
    st.write("- Medium = Some medications (11-20)")
    st.write("- High = Many medications (20+)")
    
    st.write("")
    st.write("**Blood Sugar Levels:**")
    st.write("- glucose = Glucose test results")
    st.write("- High = High glucose (max_glu_serum_3)")
    st.write("- A1C = Blood sugar test (A1Cresult_3)")
    
    st.write("")
    st.write("**Other Terms:**")
    st.write("- readmit = Patient came back to hospital")
    st.write("- gender = Male or female patient")
    st.write("- diagnos = Number of diagnoses")
    st.write("- race = Patient race category")

st.subheader("🔝 Top 10 Rules")
top_rules = get_top_rules(10)
st.dataframe(top_rules)

st.subheader("🔍 Search Rules")

# Show available keywords as buttons
st.write("**Click a keyword to search:**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("High"):
        keyword = "High"
    if st.button("Low"):
        keyword = "Low"
    
with col2:
    if st.button("medication"):
        keyword = "medication"
    if st.button("glucose"):
        keyword = "glucose"

with col3:
    if st.button("Short"):
        keyword = "Short"
    if st.button("Long"):
        keyword = "Long"

with col4:
    if st.button("readmit"):
        keyword = "readmit"
    if st.button("diagnos"):
        keyword = "diagnos"

st.write("**Or type your own:**")
keyword_input = st.text_input("Search keyword", value="", key="search_box")

# Use typed keyword if provided, otherwise use button click
if keyword_input:
    keyword = keyword_input
elif 'keyword' not in locals():
    keyword = None

if keyword:
    results = search_rules(keyword)
    
    if results.empty:
        st.warning("No rules found. Try: High, Low, medication")
    else:
        st.success("Found rules!")
        st.dataframe(results)
        
        # Simple explanation of first result
        if len(results) > 0:
            st.write("---")
            st.write("**What does the rule mean?**")
            
            confidence = results.iloc[0]['confidence']
            support = results.iloc[0]['support']
            
            st.write("IF conditions:")
            st.code(results.iloc[0]['antecedents'])
            
            st.write("THEN outcome:")
            st.code(results.iloc[0]['consequents'])
            
            # Simple reliability check
            if confidence >= 0.75:
                st.write("✅ This happens 75% of the time - Very reliable!")
            elif confidence >= 0.60:
                st.write("⚠️ This happens 60-75% of the time - Pretty reliable")
            else:
                st.write("⚠️ This happens less than 60% of the time")

