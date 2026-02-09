# Import libraries we need
import streamlit as st  # For creating the web dashboard
import pandas as pd  # For handling data
import plotly.express as px  # For creating charts

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Healthcare AI Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main background - subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e6f0fa 25%, #dfe9f3 50%, #f5f9fc 75%, #ffffff 100%);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(59, 130, 246, 0.08);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        padding: 10px 20px;
        color: #475569;
        font-weight: 600;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        border: none;
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1e40af;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        color: #64748b;
    }
    
    /* Card/container styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(100, 116, 139, 0.15);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* Info banner styling */
    .stAlert {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.12) 0%, rgba(147, 197, 253, 0.12) 100%);
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #1e40af;
        font-weight: 500;
    }
    
    /* Subheader styling */
    .stMarkdown h3 {
        color: #334155;
        font-weight: 700;
    }
    
    /* Title styling */
    .stMarkdown h1 {
        color: #1e293b;
        font-weight: 800;
    }
    
    /* Subheader in tabs */
    .stMarkdown h2 {
        color: #1e40af;
        font-weight: 700;
    }
    
    /* Summary statistics styling */
    .stMarkdown h4 {
        color: #475569;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 16px rgba(100, 116, 139, 0.12);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    /* Fix chart overflow issues */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 16px rgba(100, 116, 139, 0.12);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    /* Ensure proper chart rendering */
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    /* Fix column spacing */
    div[data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
    }
    
    /* Text content styling */
    .stMarkdown p {
        color: #475569;
    }
    
    /* Footer styling */
    .stMarkdown div[style*="text-align: center"] {
        background: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD ALL DATASETS
# ============================================
@st.cache_data
def load_diabetic_data():
    """Load the diabetic dataset used in the project"""
    df = pd.read_csv(r"D:\HealthCare System\Data\Processed\diabetic_data_cleaned.csv")
    return df

@st.cache_data
def load_vital_data():
    """Load the vital signs dataset for LSTM model"""
    df = pd.read_csv(r"D:\HealthCare System\Data\Processed\cleaned_vital_dataset.csv")
    return df

@st.cache_data
def load_feedback_data():
    """Load the patient feedback dataset for sentiment analysis"""
    df = pd.read_csv(r"D:\HealthCare System\Data\Processed\cleaned_feedback_dataset.csv")
    return df

@st.cache_data
def load_xray_data():
    """Load the pneumonia x-ray dataset"""
    import numpy as np
    # Load npz file and extract arrays into a dictionary
    npz_file = np.load(r"D:\HealthCare System\pneumoniamnist_balanced.npz")
    data = {key: npz_file[key] for key in npz_file.files}
    npz_file.close()
    return data

# ============================================
# HEADER SECTION
# ============================================
st.title("🏥 Healthcare Analytics Dashboard")
st.markdown("### Multi-Dataset Analytics and AI-Powered Predictions")

# Info banner
st.info("👋 Welcome! This dashboard shows insights from 4 different healthcare datasets used across AI models.")

st.divider()

# ============================================
# DATASET TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Diabetic Patients", "💓 Vital Signs", "💬 Patient Feedback", "🩻 X-Ray Images"])

# ============================================
# TAB 1: DIABETIC PATIENTS
# ============================================
with tab1:
    # Load diabetic data
    df = load_diabetic_data()
    
    st.subheader("📊 Diabetic Patient Dataset")
    st.write("**Used in:** Readmission Prediction, Length of Stay, Clustering, Association Rules")
    
    # Calculate metrics from data
    total_patients = len(df)
    avg_age_range = df['age'].mode()[0]  # Most common age range
    male_count = len(df[df['gender'] == 1])  # 1 = Male
    female_count = len(df[df['gender'] == 0])  # 0 = Female
    male_percent = (male_count / total_patients) * 100
    readmission_rate = (df['readmitted_binary'].sum() / total_patients) * 100

    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # Metric 1: Total Patients
    with col1:
        st.metric(
            label="👥 Total Patients",
            value=f"{total_patients:,}",
            delta="Diabetic Cases"
        )

    # Metric 2: Most Common Age Range
    with col2:
        st.metric(
            label="📅 Most Common Age",
            value=avg_age_range,
            delta="Age Range"
        )

    # Metric 3: Gender Split
    with col3:
        st.metric(
            label="👨 Male Patients",
            value=f"{male_percent:.1f}%",
            delta=f"{male_count:,} patients"
        )

    # Metric 4: Readmission Rate
    with col4:
        st.metric(
            label="🔄 Readmission Rate",
            value=f"{readmission_rate:.1f}%",
            delta="30-Day Readmitted"
        )

    st.markdown("---")

    # Charts Row 1
    st.markdown("#### Patient Demographics")
    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Age Range Distribution
    with chart_col1:
        age_counts = df['age'].value_counts().sort_index()
        fig_age = px.bar(
            x=age_counts.index,
            y=age_counts.values,
            title="Age Range Distribution",
            labels={'x': 'Age Range', 'y': 'Number of Patients'},
            color_discrete_sequence=['#4ECDC4']
        )
        fig_age.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Chart 2: Gender Distribution
    with chart_col2:
        gender_counts = df['gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        fig_gender.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    # Charts Row 2
    st.markdown("#### Hospital Metrics")
    chart_col3, chart_col4 = st.columns(2)

    # Chart 3: Readmission Status
    with chart_col3:
        readmit_labels = {0: 'No Readmission', 1: 'Readmitted'}
        df['readmit_label'] = df['readmitted_binary'].map(readmit_labels)
        readmit_counts = df['readmit_label'].value_counts()
        fig_readmit = px.pie(
            values=readmit_counts.values,
            names=readmit_counts.index,
            title="30-Day Readmission Status",
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig_readmit.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_readmit, use_container_width=True)

    # Chart 4: Time in Hospital Distribution
    with chart_col4:
        fig_los = px.histogram(
            df,
            x='time_in_hospital',
            title="Length of Stay Distribution",
            labels={'time_in_hospital': 'Days in Hospital', 'count': 'Number of Patients'},
            color_discrete_sequence=['#95E1D3']
        )
        fig_los.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_los, use_container_width=True)

    st.markdown("---")

    # Summary Statistics
    st.markdown("#### 📋 Summary Statistics")
    sum_col1, sum_col2, sum_col3 = st.columns(3)

    with sum_col1:
        st.markdown("**Patient Demographics**")
        st.write(f"• Total Patients: {total_patients:,}")
        st.write(f"• Age Ranges: {df['age'].nunique()} groups")
        st.write(f"• Male: {male_count:,} ({male_percent:.1f}%)")
        st.write(f"• Female: {female_count:,} ({100-male_percent:.1f}%)")

    with sum_col2:
        st.markdown("**Hospital Metrics**")
        avg_los = df['time_in_hospital'].mean()
        avg_meds = df['num_medications'].mean()
        avg_procedures = df['num_procedures'].mean()
        st.write(f"• Avg Length of Stay: {avg_los:.1f} days")
        st.write(f"• Avg Medications: {avg_meds:.1f}")
        st.write(f"• Avg Procedures: {avg_procedures:.1f}")

    with sum_col3:
        st.markdown("**Readmission Analytics**")
        readmitted_count = df['readmitted_binary'].sum()
        not_readmitted = total_patients - readmitted_count
        st.write(f"• Readmitted: {readmitted_count:,} ({readmission_rate:.1f}%)")
        st.write(f"• Not Readmitted: {not_readmitted:,}")
        st.write(f"• Total Records: {len(df):,}")

# ============================================
# TAB 2: VITAL SIGNS
# ============================================
with tab2:
    # Load vital signs data
    vital_df = load_vital_data()
    
    st.subheader("💓 Vital Signs Dataset")
    st.write("**Used in:** LSTM ICU Risk Prediction Model")
    
    # Calculate metrics
    total_records = len(vital_df)
    unique_patients = vital_df['Patient ID'].nunique()
    high_risk = len(vital_df[vital_df['Risk Category_High Risk'] == 1])
    low_risk = len(vital_df[vital_df['Risk Category_Low Risk'] == 1])
    high_risk_percent = (high_risk / total_records) * 100
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Records",
            value=f"{total_records:,}",
            delta="Vital Sign Readings"
        )
    
    with col2:
        st.metric(
            label="👥 Unique Patients",
            value=f"{unique_patients:,}",
            delta="ICU Patients"
        )
    
    with col3:
        st.metric(
            label="⚠️ High Risk",
            value=f"{high_risk_percent:.1f}%",
            delta=f"{high_risk:,} readings"
        )
    
    with col4:
        avg_hr = vital_df['Heart Rate'].mean()
        st.metric(
            label="❤️ Avg Heart Rate",
            value=f"{avg_hr:.0f} bpm",
            delta="Across all patients"
        )
    
    st.markdown("---")
    
    # Charts
    st.markdown("#### Vital Sign Distributions")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_hr = px.histogram(
            vital_df,
            x='Heart Rate',
            title="Heart Rate Distribution",
            labels={'Heart Rate': 'Heart Rate (bpm)', 'count': 'Frequency'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_hr.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_hr, use_container_width=True)
    
    with chart_col2:
        risk_data = pd.DataFrame({
            'Risk Level': ['High Risk', 'Low Risk'],
            'Count': [high_risk, low_risk]
        })
        fig_risk = px.pie(
            risk_data,
            values='Count',
            names='Risk Level',
            title="Risk Category Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        fig_risk.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        fig_temp = px.histogram(
            vital_df,
            x='Body Temperature',
            title="Body Temperature Distribution",
            labels={'Body Temperature': 'Temperature (°C)', 'count': 'Frequency'},
            color_discrete_sequence=['#FFA07A']
        )
        fig_temp.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with chart_col4:
        fig_o2 = px.histogram(
            vital_df,
            x='Oxygen Saturation',
            title="Oxygen Saturation Distribution",
            labels={'Oxygen Saturation': 'SpO2 (%)', 'count': 'Frequency'},
            color_discrete_sequence=['#95E1D3']
        )
        fig_o2.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_o2, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.markdown("#### 📋 Summary Statistics")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    
    with sum_col1:
        st.markdown("**Dataset Overview**")
        st.write(f"• Total Records: {total_records:,}")
        st.write(f"• Unique Patients: {unique_patients:,}")
        avg_readings = total_records / unique_patients
        st.write(f"• Avg Readings/Patient: {avg_readings:.1f}")
    
    with sum_col2:
        st.markdown("**Vital Sign Averages**")
        st.write(f"• Heart Rate: {vital_df['Heart Rate'].mean():.1f} bpm")
        st.write(f"• Resp Rate: {vital_df['Respiratory Rate'].mean():.1f} /min")
        st.write(f"• Temperature: {vital_df['Body Temperature'].mean():.1f} °C")
        st.write(f"• SpO2: {vital_df['Oxygen Saturation'].mean():.1f}%")
    
    with sum_col3:
        st.markdown("**Risk Analytics**")
        st.write(f"• High Risk: {high_risk:,} ({high_risk_percent:.1f}%)")
        st.write(f"• Low Risk: {low_risk:,} ({100-high_risk_percent:.1f}%)")
        st.write(f"• Total Readings: {total_records:,}")

# ============================================
# TAB 3: PATIENT FEEDBACK
# ============================================
with tab3:
    # Load feedback data
    feedback_df = load_feedback_data()
    
    st.subheader("💬 Patient Feedback Dataset")
    st.write("**Used in:** BERT Sentiment Analysis Model")
    
    # Calculate metrics
    total_feedback = len(feedback_df)
    positive_count = len(feedback_df[feedback_df['Sentiment'] == 1])
    negative_count = len(feedback_df[feedback_df['Sentiment'] == 0])
    positive_percent = (positive_count / total_feedback) * 100
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💬 Total Feedback",
            value=f"{total_feedback:,}",
            delta="Patient Comments"
        )
    
    with col2:
        st.metric(
            label="😊 Positive",
            value=f"{positive_percent:.1f}%",
            delta=f"{positive_count:,} comments"
        )
    
    with col3:
        st.metric(
            label="😞 Negative",
            value=f"{100-positive_percent:.1f}%",
            delta=f"{negative_count:,} comments"
        )
    
    with col4:
        avg_length = feedback_df['clean_text'].str.len().mean()
        st.metric(
            label="📝 Avg Text Length",
            value=f"{avg_length:.0f} chars",
            delta="Per feedback"
        )
    
    st.markdown("---")
    
    # Charts
    st.markdown("#### Sentiment Distribution")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [positive_count, negative_count]
        })
        fig_sentiment = px.pie(
            sentiment_data,
            values='Count',
            names='Sentiment',
            title="Sentiment Distribution",
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig_sentiment.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with chart_col2:
        text_lengths = feedback_df['clean_text'].str.len()
        fig_length = px.histogram(
            x=text_lengths,
            title="Feedback Text Length Distribution",
            labels={'x': 'Text Length (characters)', 'count': 'Frequency'},
            color_discrete_sequence=['#95E1D3']
        )
        fig_length.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.markdown("#### 📋 Summary Statistics")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    
    with sum_col1:
        st.markdown("**Dataset Overview**")
        st.write(f"• Total Feedback: {total_feedback:,}")
        st.write(f"• Avg Text Length: {avg_length:.0f} chars")
        st.write(f"• Min Length: {text_lengths.min()} chars")
        st.write(f"• Max Length: {text_lengths.max()} chars")
    
    with sum_col2:
        st.markdown("**Sentiment Breakdown**")
        st.write(f"• Positive: {positive_count:,} ({positive_percent:.1f}%)")
        st.write(f"• Negative: {negative_count:,} ({100-positive_percent:.1f}%)")
        ratio = positive_count / negative_count if negative_count > 0 else 0
        st.write(f"• Pos/Neg Ratio: {ratio:.2f}")
    
    with sum_col3:
        st.markdown("**Sample Feedback**")
        sample_positive = feedback_df[feedback_df['Sentiment'] == 1].sample(1)['clean_text'].values[0]
        sample_negative = feedback_df[feedback_df['Sentiment'] == 0].sample(1)['clean_text'].values[0]
        st.write(f"**Positive:** _{sample_positive[:50]}..._")
        st.write(f"**Negative:** _{sample_negative[:50]}..._")

# ============================================
# TAB 4: X-RAY IMAGES
# ============================================
with tab4:
    # Load x-ray data
    xray_data = load_xray_data()
    
    st.subheader("🩻 X-Ray Image Dataset")
    st.write("**Used in:** CNN Pneumonia Detection Model")
    
    # Extract dataset info
    train_images = xray_data['images']  # Training images
    train_labels = xray_data['labels']  # Training labels
    val_images = xray_data['val_images']  # Validation images
    val_labels = xray_data['val_labels']  # Validation labels
    test_images = xray_data['test_images']  # Test images
    test_labels = xray_data['test_labels']  # Test labels
    
    total_images = len(train_images) + len(val_images) + len(test_images)
    train_pneumonia = (train_labels == 1).sum()
    val_pneumonia = (val_labels == 1).sum()
    test_pneumonia = (test_labels == 1).sum()
    total_pneumonia = train_pneumonia + val_pneumonia + test_pneumonia
    pneumonia_percent = (total_pneumonia / total_images) * 100
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🖼️ Total Images",
            value=f"{total_images:,}",
            delta="All Splits Combined"
        )
    
    with col2:
        st.metric(
            label="📚 Training Set",
            value=f"{len(train_images):,}",
            delta="For Model Learning"
        )
    
    with col3:
        st.metric(
            label="✅ Validation Set",
            value=f"{len(val_images):,}",
            delta="For Tuning"
        )
    
    with col4:
        st.metric(
            label="🧪 Test Set",
            value=f"{len(test_images):,}",
            delta="Final Evaluation"
        )
    
    st.markdown("---")
    
    # Additional info row
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric(
            label="🦠 Total Pneumonia Cases",
            value=f"{total_pneumonia:,}",
            delta=f"{pneumonia_percent:.1f}% of dataset"
        )
    
    with info_col2:
        st.metric(
            label="✨ Normal Cases",
            value=f"{total_images - total_pneumonia:,}",
            delta=f"{100-pneumonia_percent:.1f}% of dataset"
        )
    
    with info_col3:
        normal_to_pneumonia_ratio = (total_images - total_pneumonia) / total_pneumonia
        balance_status = "Balanced ✓" if 40 <= pneumonia_percent <= 60 else "Imbalanced ⚠️"
        st.metric(
            label="📊 Dataset Balance",
            value=balance_status,
            delta=f"Pneumonia:Normal = 1:{normal_to_pneumonia_ratio:.2f}"
        )
    
    st.markdown("---")
    
    # Charts
    st.markdown("#### Dataset Distribution")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        split_data = pd.DataFrame({
            'Split': ['Training', 'Validation', 'Test'],
            'Count': [len(train_images), len(val_images), len(test_images)]
        })
        fig_split = px.bar(
            split_data,
            x='Split',
            y='Count',
            title="Train/Val/Test Split",
            labels={'Count': 'Number of Images'},
            color_discrete_sequence=['#4ECDC4'],
            text='Count'
        )
        fig_split.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_split.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=60, b=60),
            yaxis_title="Number of Images",
            xaxis_title="Dataset Split"
        )
        st.plotly_chart(fig_split, use_container_width=True)
    
    with chart_col2:
        label_data = pd.DataFrame({
            'Label': ['Normal', 'Pneumonia'],
            'Count': [total_images - total_pneumonia, total_pneumonia]
        })
        fig_labels = px.pie(
            label_data,
            values='Count',
            names='Label',
            title="Label Distribution",
            color_discrete_sequence=['#4ECDC4', '#FF6B6B']
        )
        fig_labels.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        fig_labels.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_labels, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.markdown("#### 📋 Summary Statistics")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    
    with sum_col1:
        st.markdown("**Dataset Overview**")
        st.write(f"• Total Images: {total_images:,}")
        st.write(f"• Training Images: {len(train_images):,}")
        st.write(f"• Validation Images: {len(val_images):,}")
        st.write(f"• Test Images: {len(test_images):,}")
    
    with sum_col2:
        st.markdown("**Training Set**")
        train_normal = len(train_images) - train_pneumonia
        st.write(f"• Normal: {train_normal:,}")
        st.write(f"• Pneumonia: {train_pneumonia:,}")
        train_pneumonia_pct = (train_pneumonia / len(train_images)) * 100
        st.write(f"• Pneumonia %: {train_pneumonia_pct:.1f}%")
    
    with sum_col3:
        st.markdown("**Test Set**")
        test_normal = len(test_images) - test_pneumonia
        st.write(f"• Normal: {test_normal:,}")
        st.write(f"• Pneumonia: {test_pneumonia:,}")
        test_pneumonia_pct = (test_pneumonia / len(test_images)) * 100
        st.write(f"• Pneumonia %: {test_pneumonia_pct:.1f}%")

st.divider()

# ============================================
# AI MODELS QUICK ACCESS
# ============================================
st.subheader("🤖 AI Prediction Models")

st.markdown("""
**Use the sidebar to access prediction models:**
- **Readmission Risk:** Predict 30-day readmission probability
- **Length of Stay:** Estimate hospital stay duration
- **Patient Clustering:** Group similar patients
- **Sentiment Analysis:** Analyze patient feedback
- **Association Rules:** Find disease patterns
- **X-Ray Diagnostics:** Detect pneumonia from images
- **ICU Risk:** Predict patient deterioration
""")

st.divider()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏥 Healthcare Analytics Dashboard | Built with Streamlit & Python</p>
        <p>Multi-dataset insights for comprehensive patient care analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)
