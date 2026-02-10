import tensorflow as tf
import numpy as np

# ---------------------------------------------------
# Load the saved CNN model
# ---------------------------------------------------
model_path= r"D:\HealthCare System\notebooks\deep learning\CNN\resnet_pneumonia_model.h5"
model = tf.keras.models.load_model(model_path)
def preprocess_image(image):
    """
    Converts uploaded PIL image into model-ready format:
    - RGB conversion
    - Resize to 224x224
    - Normalize to 0–1
    - Add batch dimension
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# ---------------------------------------------------
# Predict pneumonia vs normal
# ---------------------------------------------------
def predict_pneumonia(image, threshold=0.5132):
    """
    Returns:
    - label: 'Pneumonia' or 'Normal'
    - probability: model confidence score
    """
    img = preprocess_image(image)
    prob = model.predict(img)[0][0]

    label = "Pneumonia" if prob >= threshold else "Normal"

    return label, float(prob)
