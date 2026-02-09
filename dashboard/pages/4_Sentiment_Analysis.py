# Import libraries
import streamlit as st  # For creating the web dashboard
import sys  # To add custom folder paths
import importlib  # To reload modules

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fef3f8 0%, #f0e8ff 50%, #ffffff 100%);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 248, 255, 0.95) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(221, 214, 254, 0.8);
    }
    .stButton>button {
        background: linear-gradient(90deg, #a855f7 0%, #c084fc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid rgba(221, 214, 254, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add the Models folder so Python can find our sentiment_model.py file
sys.path.insert(0, r"D:\HealthCare System\notebooks\Models")

# Import our sentiment prediction functions
import sentiment_model
importlib.reload(sentiment_model)  # Force reload to get latest changes
from sentiment_model import predict_sentiment, predict_batch

# Page title
st.title("💬 Patient Feedback Sentiment Analysis")

# Key use case banner
st.info("🎯 **Use Case:** Sentiment model detects dissatisfaction trends for hospital QA teams.")

st.write("Uses fine-tuned BERT model to analyze patient feedback sentiment")

# ---------------------------------------------------------
# OPTION A: ANALYZE ONE FEEDBACK AT A TIME
# ---------------------------------------------------------
st.subheader("Option A: Analyze Single Feedback")

# Create example texts for testing
positive_example = "The staff was incredibly caring and the treatment was excellent. I felt well taken care of throughout my stay."
negative_example = "Long wait times, rude staff, and poor communication. Very disappointing experience."
mixed_example = "The doctors were great but the waiting room was crowded and uncomfortable."

# Show quick example buttons
st.write("**Quick Examples:**")
col1, col2, col3 = st.columns(3)

# Use session state to remember the selected text
if 'selected_text' not in st.session_state:
    st.session_state.selected_text = ""

# Button 1: Positive example
with col1:
    if st.button("😊 Positive Example"):
        st.session_state.selected_text = positive_example

# Button 2: Negative example
with col2:
    if st.button("😞 Negative Example"):
        st.session_state.selected_text = negative_example

# Button 3: Mixed example
with col3:
    if st.button("😐 Mixed Example"):
        st.session_state.selected_text = mixed_example

# Text box where user can type feedback
user_text = st.text_area(
    "Enter patient feedback:",
    value=st.session_state.selected_text,  # Use saved value from session state
    height=150,
    placeholder="Type or paste patient feedback here..."
)

# Button to analyze the text
if st.button("🔍 Analyze Sentiment", type="primary"):
    
    # Check if user entered something
    if user_text.strip():
        
        # Show loading message
        with st.spinner("Analyzing sentiment..."):
            # Call our prediction function
            sentiment, positive_score, negative_score = predict_sentiment(user_text)
        
        # Check for mixed sentiment indicators
        mixed_words = ["but", "however", "although", "though", "yet", "while", "despite"]
        has_mixed_indicator = any(word in user_text.lower() for word in mixed_words)
        
        # Show the result
        st.subheader("📊 Result")
        
        # Show different colors based on sentiment
        if sentiment == "Positive":
            st.success(f"**Sentiment:** {sentiment} 😊")
        else:
            st.error(f"**Sentiment:** {sentiment} 😞")
        
        # Show BOTH scores so user can see the full picture
        col1, col2 = st.columns(2)
        with col1:
            st.metric("😊 Positive Score", f"{positive_score:.1%}")
            st.progress(positive_score)
        with col2:
            st.metric("😞 Negative Score", f"{negative_score:.1%}")
            st.progress(negative_score)
        
        # Explain what the confidence means
        confidence = max(positive_score, negative_score)
        
        # Check for mixed sentiment
        if has_mixed_indicator:
            st.warning("⚠️ **Mixed Sentiment Detected** - Text contains contrasting opinions (words like 'but', 'however'). Model may focus on dominant sentiment only.")
        elif confidence > 0.9:
            st.write("✅ **Very confident prediction**")
        elif confidence > 0.7:
            st.write("✅ **Confident prediction**")
        else:
            st.write("⚠️ **Moderate confidence - feedback may be mixed**")
    else:
        # If no text entered, show warning
        st.warning("⚠️ Please enter some text to analyze")

# Divider line
st.divider()

# ---------------------------------------------------------
# OPTION B: ANALYZE MULTIPLE FEEDBACKS AT ONCE
# ---------------------------------------------------------
st.subheader("Option B: Analyze Multiple Feedbacks")

st.write("Enter multiple feedbacks (one per line):")

# Text area for multiple feedbacks
batch_text = st.text_area(
    "Multiple feedbacks:",
    height=200,
    placeholder="Example:\nThe nurses were very attentive and kind.\nTerrible wait times and unhelpful staff.\nClean facilities but expensive."
)

# Button to analyze batch
if st.button("🔍 Analyze Batch", type="primary"):
    
    # Check if user entered something
    if batch_text.strip():
        
        # Split the text into separate lines
        feedbacks = []
        for line in batch_text.split('\n'):
            # Remove extra spaces
            clean_line = line.strip()
            # Only add non-empty lines
            if clean_line:
                feedbacks.append(clean_line)
        
        # If we have valid feedbacks
        if feedbacks:
            
            # Show loading message
            with st.spinner(f"Analyzing {len(feedbacks)} feedbacks..."):
                # Analyze all feedbacks
                results = predict_batch(feedbacks)
            
            st.subheader("📊 Batch Results")
            
            # Count positive and negative
            positive_count = 0
            negative_count = 0
            total_positive_score = 0
            total_negative_score = 0
            
            for sentiment, pos_score, neg_score in results:
                if sentiment == "Positive":
                    positive_count = positive_count + 1
                else:
                    negative_count = negative_count + 1
                total_positive_score = total_positive_score + pos_score
                total_negative_score = total_negative_score + neg_score
            
            # Calculate average scores
            avg_positive = total_positive_score / len(results)
            avg_negative = total_negative_score / len(results)
            
            # Show summary in 3 columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😊 Positive", positive_count)
            with col2:
                st.metric("😞 Negative", negative_count)
            with col3:
                st.metric("📊 Avg Positive Score", f"{avg_positive:.1%}")
            
            st.write("---")
            
            # Show detailed results for each feedback
            feedback_number = 1
            for i in range(len(feedbacks)):
                feedback = feedbacks[i]
                sentiment, positive_score, negative_score = results[i]
                
                # Check for mixed sentiment indicators
                mixed_words = ["but", "however", "although", "though", "yet", "while", "despite"]
                has_mixed_indicator = any(word in feedback.lower() for word in mixed_words)
                
                # Create expandable section for each feedback
                with st.expander(f"Feedback {feedback_number}: {sentiment} (😊 {positive_score:.1%} / 😞 {negative_score:.1%})"):
                    st.write(f"**Text:** {feedback}")
                    
                    # Show sentiment with color
                    if sentiment == "Positive":
                        st.success(f"Sentiment: {sentiment} 😊")
                    else:
                        st.error(f"Sentiment: {sentiment} 😞")
                    
                    # Show both scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"😊 Positive: {positive_score:.1%}")
                        st.progress(positive_score)
                    with col2:
                        st.write(f"😞 Negative: {negative_score:.1%}")
                        st.progress(negative_score)
                    
                    # Show mixed sentiment warning
                    if has_mixed_indicator:
                        st.warning("⚠️ **Mixed Sentiment** - Contains contrasting opinions")
                
                feedback_number = feedback_number + 1
        else:
            st.warning("⚠️ No valid feedbacks found")
    else:
        st.warning("⚠️ Please enter feedbacks to analyze")

# Divider line
st.divider()

# ---------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------
with st.expander("ℹ️ About This Model"):
    st.write("""
    **Model:** BERT (Bidirectional Encoder Representations from Transformers)
    
    **Training:** Fine-tuned on patient feedback dataset
    
    **Use Cases:**
    - Analyze patient satisfaction surveys
    - Monitor feedback trends over time
    - Identify areas needing improvement
    - Prioritize responses to negative feedback
    
    **Confidence Scores:**
    - >90%: Very confident prediction
    - 70-90%: Confident prediction
    - <70%: Moderate confidence (possibly mixed sentiment)
    """)
