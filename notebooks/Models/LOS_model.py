import joblib
import numpy as np

# -----------------------------
# LOAD TRAINED ARTIFACTS
# -----------------------------

# Load the tuned XGBoost regression model
model = joblib.load(r"D:\HealthCare System\Trained_Models\LOS_Prediction\los_model.pkl")

# Load the feature order used during training
feature_order = joblib.load(r"D:\HealthCare System\Trained_Models\LOS_Prediction\los_features.pkl")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_los(input_dict):
    """
    input_dict example:
    {
        "num_medications": 10,
        "num_lab_procedures": 35,
        "number_diagnoses": 5,
        "num_procedures": 1,
        "discharge_disposition_id": 1,
        "age": 60
    }

    Steps:
    1. Arrange inputs in the same order as training
    2. Convert to numpy array
    3. Predict LOS using the trained model
    """

    # Convert dictionary → array in correct order
    X = np.array([[input_dict[feat] for feat in feature_order]])

    # Predict LOS (in days)
    pred = model.predict(X)[0]

    # Return as float
    return float(pred)
