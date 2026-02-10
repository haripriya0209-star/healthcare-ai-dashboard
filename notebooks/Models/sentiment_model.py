# Import libraries we need
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # BERT model tools
import torch  # PyTorch for deep learning

# -----------------------------
# STEP 1: LOAD THE TRAINED MODEL
# -----------------------------

# This is where our trained BERT model is saved
model_path = r"D:\HealthCare System\notebooks\deep learning\bert_sentiment\checkpoint-135"

# Load the tokenizer (converts text into numbers that BERT understands)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the trained model (this is the AI brain that predicts sentiment)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put model in evaluation mode (not training mode)
model.eval()


# -----------------------------
# STEP 2: FUNCTION TO PREDICT SENTIMENT
# -----------------------------

def predict_sentiment(text):
    """
    This function takes patient feedback text and tells you if it's Positive or Negative.
    
    Example:
        text = "The doctors were very helpful"
        Result: ("Positive", 0.95, 0.05) means 95% positive, 5% negative
    """
    
    # STEP 2A: Convert text into numbers (tokenization)
    # BERT doesn't understand words, it needs numbers
    inputs = tokenizer(
        text,                    # The patient feedback text
        return_tensors="pt",     # Return as PyTorch tensor
        padding=True,            # Make all sentences same length
        truncation=True,         # Cut text if too long
        max_length=128           # Maximum 128 words
    )
    
    # STEP 2B: Get the model's prediction
    with torch.no_grad():  # Don't calculate gradients (we're not training)
        
        # Feed the numbers into BERT model
        outputs = model(**inputs)
        
        # Get the raw scores (called logits)
        logits = outputs.logits
        
        # Convert raw scores into probabilities (0 to 1)
        probs = torch.softmax(logits, dim=1)
        
        # Get both scores (model labels: 0=Negative, 1=Positive based on training data)
        negative_score = probs[0][0].item()  # Probability at index 0 = Negative
        positive_score = probs[0][1].item()  # Probability at index 1 = Positive
        
        # Find which class has highest score
        pred_class = logits.argmax(dim=1).item()
    
    # STEP 2C: Convert prediction number to label
    # Model training used: 0=Negative, 1=Positive
    if pred_class == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    
    # Return sentiment, positive score, and negative score
    return sentiment, positive_score, negative_score


# -----------------------------
# STEP 3: FUNCTION FOR MULTIPLE FEEDBACKS
# -----------------------------

def predict_batch(texts):
    """
    Analyze multiple patient feedbacks at once.
    
    Example:
        texts = ["Great service", "Terrible wait times", "Very clean"]
        Result: [("Positive", 0.92, 0.08), ("Negative", 0.12, 0.88), ("Positive", 0.85, 0.15)]
    """
    
    # Create empty list to store results
    results = []
    
    # Loop through each feedback text
    for text in texts:
        # Get sentiment for this one feedback
        sentiment, positive_score, negative_score = predict_sentiment(text)
        
        # Add the result to our list
        results.append((sentiment, positive_score, negative_score))
    
    # Return all the results
    return results
