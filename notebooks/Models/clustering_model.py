import joblib
import numpy as np

# -----------------------------
# LOAD TRAINED ARTIFACTS
# -----------------------------

# Load KMeans model trained on PCA components
model = joblib.load(r"D:\HealthCare System\Trained_Models\Clustering\cluster_model.pkl")

# Load the scaler used before PCA
scaler = joblib.load(r"D:\HealthCare System\Trained_Models\Clustering\cluster_scaler.pkl")

# Load PCA transformer
pca = joblib.load(r"D:\HealthCare System\Trained_Models\Clustering\cluster_pca.pkl")

# Load feature order used during training
feature_order = joblib.load(r"D:\HealthCare System\Trained_Models\Clustering\cluster_features.pkl")


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_cluster(input_dict):
    """
    input_dict example:
    {
        "age": 60,
        "num_medications": 10,
        "num_lab_procedures": 40,
        "number_diagnoses": 5,
        "time_in_hospital": 4
    }

    Steps:
    1. Arrange inputs in the same order as training
    2. Scale using saved scaler
    3. Apply PCA transformation
    4. Predict cluster using trained KMeans
    """

    # Convert dictionary → array in correct order
    X = np.array([[input_dict[feat] for feat in feature_order]])

    # Scale input
    X_scaled = scaler.transform(X)

    # PCA transform
    X_pca = pca.transform(X_scaled)

    # Predict cluster
    cluster = model.predict(X_pca)[0]

    return int(cluster)


# -----------------------------
# OPTIONAL: CLUSTER PROFILES
# -----------------------------

cluster_profiles = {
    0: "Older patients with high inpatient visits (2+), moderate complexity, stable readmission",
    1: "Low-risk: minimal hospital use, shortest stays, lowest medications",
    2: "High-complexity: oldest age, longest stays (8+ days), highest meds/labs/procedures",
    3: "Older chronic-care: high meds/labs but very low hospital visit frequency",
    4: "Younger with extremely high utilization: frequent inpatient (5+), ER (4+), outpatient visits"
}

def get_cluster_profile(cluster_id):
    return cluster_profiles.get(cluster_id, "Cluster profile not defined")
