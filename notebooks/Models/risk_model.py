import joblib          # used to load .pkl files (saved model + scaler)
import numpy as np     # used to build numeric arrays for prediction


# -----------------------------
# LOAD  TRAINED ARTIFACTS
# -----------------------------

# Load the trained XGBoost model  saved from the notebook
model = joblib.load(r"D:\HealthCare System\notebooks\risk_model.pkl")
   

# Load the StandardScaler used during training
# This ensures new inputs are scaled the same way
scaler = joblib.load(r"D:\HealthCare System\notebooks\risk_scaler.pkl")


# Load the exact feature order used during training
# This is CRITICAL — model expects inputs in this order
feature_order = joblib.load(r"D:\HealthCare System\notebooks\risk_features.pkl")  


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_risk(input_dict):
    """
    input_dict is a dictionary coming from Streamlit UI.
    Example:
    {
        "age": 50,
        "bmi": 28.5,
        "systolic_bp": 130,
        "glucose": 110,
        "diabetes": 1
    }

    This function:
    1. Converts the dictionary into a numpy array
    2. Arranges values in the SAME order as training
    3. Scales the values using the saved StandardScaler
    4. Runs the model prediction
    5. Returns (class_label, probability)
    """

    # -----------------------------------------
    # Convert dictionary → array in correct order
    # -----------------------------------------
    #  loop through feature_order to ensure correct column sequence

    X = np.array([[input_dict.get(feat,0) for feat in feature_order]])

    # -----------------------------------------
    # Scale the input using the SAME scaler used in training
    # -----------------------------------------
    X_scaled = scaler.transform(X)

    # -----------------------------------------
    # Predict class (0 or 1)
    # -----------------------------------------
    pred = model.predict(X_scaled)[0]

    # -----------------------------------------
    # Predict probability of the predicted class
    # -----------------------------------------
    prob = model.predict_proba(X_scaled)[0].max()

    # Convert numpy types → normal Python types
    return int(pred), float(prob)
