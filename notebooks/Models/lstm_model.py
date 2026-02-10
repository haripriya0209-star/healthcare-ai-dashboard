import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# Load the saved LSTM model
# ---------------------------------------------------

model_path=r"D:\HealthCare System\notebooks\deep learning\LSTM\lstm_risk_model.h5"
lstm_model = tf.keras.models.load_model(model_path)

# ---------------------------------------------------
# Load and fit scaler (model was trained on normalized data)
# ---------------------------------------------------

df = pd.read_csv(r"D:\HealthCare System\Data\Processed\cleaned_vital_dataset.csv")
features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure',
    'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_MAP'
]
scaler = StandardScaler()
scaler.fit(df[features])

# ---------------------------------------------------
# Predict risk from time-series vitals
# ---------------------------------------------------

# def predict_risk(sequence,threshold=0.5):
#     """
#     Input:
#     - sequence: NumPy array of shape (timesteps, features)

#     Returns:
#     - label: 'High Risk' or 'Low Risk'
#     - probability: model confidence score
#     """
#     sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension  
#     prob=lstm_model.predict(sequence)[0][0]
#     label = "High Risk" if prob >= threshold else "Low Risk"
#     return label, float(prob)

def predict_risk(sequence, threshold=0.5, temperature=2.0):
    '''
    Predict risk with temperature scaling for calibration.
    
    Args:
        sequence: NumPy array of raw vital signs (timesteps, 9 features)
        threshold: Decision boundary (default 0.5)
        temperature: Calibration factor (default 2.0)
    
    Returns:
        label: "High Risk" or "Low Risk"
        probability: Calibrated confidence (0-1)
    '''
    # STEP 1: Normalize the vitals (model was trained on scaled data)
    sequence_scaled = scaler.transform(sequence)
    
    # STEP 2: Add batch dimension if needed (3, 9) -> (1, 3, 9)
    if len(sequence_scaled.shape) == 2:
        sequence_scaled = np.expand_dims(sequence_scaled, axis=0)
    
    # STEP 3: Get raw prediction from model
    y_pred_prob_raw = lstm_model.predict(sequence_scaled, verbose=0)
    
    # STEP 4: Apply temperature scaling for calibration
    logits = tf.math.log(y_pred_prob_raw / (1 - y_pred_prob_raw + 1e-7))
    y_pred_prob = float(tf.nn.sigmoid(logits / temperature).numpy()[0][0])
    
    # STEP 5: Determine label
    label = "High Risk" if y_pred_prob >= threshold else "Low Risk"
    
    return label, y_pred_prob