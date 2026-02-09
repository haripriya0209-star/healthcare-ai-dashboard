import streamlit as st
from PIL import Image
import sys

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 253, 250, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(20, 184, 166, 0.15);
        border: 1px solid rgba(204, 251, 241, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #14b8a6 0%, #2dd4bf 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Add Models folder using absolute path
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

from cnn_model import predict_pneumonia

st.title("🩻 Pneumonia Detection using ResNet50")

st.write("""
Upload a chest X-ray image to detect whether the model predicts **Pneumonia** or **Normal**.
This model is a fine‑tuned ResNet50 trained on the PneumoniaMNIST dataset.
""")

uploaded = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded X-ray", width=350)

    label, prob = predict_pneumonia(image)

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence Score: **{prob:.3f}**")
    
    # Show confidence level warning
    if prob < 0.52 and prob > 0.48:
        st.warning("⚠️ **Low Confidence**: The model is uncertain (score close to 0.5). Results may not be reliable.")

    # Clinical interpretation box
    if label == "Pneumonia":
        st.info("""
        **Interpretation:**  
        The model predicts *Pneumonia*.  
        This suggests the X-ray shows patterns commonly associated with lung infection, such as opacities or consolidation.  
        Please note: This is an AI prediction and should be reviewed by a radiologist.
        """)
    else:
        st.success("""
        **Interpretation:**  
        The model predicts *Normal*.  
        No strong pneumonia-like patterns were detected.  
        This does not replace clinical evaluation.
        """)
