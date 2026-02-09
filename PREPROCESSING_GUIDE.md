# 📊 Preprocessing Guide - Healthcare AI Dashboard

## 📋 Overview
This document maps preprocessing notebooks to their respective AI models, explaining what data transformations were applied for each use case.

---

## 🗂️ Preprocessing Structure

### **Notebook 1: preprocessing_version_1.ipynb**
**Purpose:** Handles ALL preprocessing for diabetic patient tabular data

**Dataset:** `diabetic_data.csv` (101,766 diabetic hospital records)

**Models Using This Preprocessing:**
1. ✅ **Readmission Risk (Classification)** - XGBoost model
2. ✅ **Length of Stay (Regression)** - XGBoost regressor
3. ✅ **Patient Segmentation (Clustering)** - KMeans clustering
4. ✅ **Association Rules (Pattern Mining)** - Apriori algorithm

**Key Preprocessing Steps:**
```
1. Data Loading & Exploration
   - Load diabetic_data.csv
   - Check shape: (101,766 rows, 50 columns)
   - Identify data types (numerical, categorical)

2. Missing Value Handling
   - Identify columns with missing values
   - Impute numerical: mean/median strategy
   - Impute categorical: mode strategy
   - Drop rows with excessive missing values

3. Categorical Encoding
   - Label Encoding for ordinal variables (age ranges)
   - One-Hot Encoding for nominal variables (race, gender)
   - Binary encoding for diagnosis codes

4. Feature Engineering
   - Create readmitted_binary (0/1 from readmitted column)
   - Bin time_in_hospital into categories (Short/Medium/Long)
   - Aggregate medication features
   - Create diagnosis groupings

5. Feature Scaling
   - StandardScaler for classification/regression models
   - Min-Max scaling for clustering (0-1 range)
   - Normalize for better convergence

6. Feature Selection
   - Correlation analysis (remove multicollinearity)
   - Variance threshold (remove low-variance features)
   - Mutual information scores for importance

7. Data Splitting
   - Train/Test split (80/20)
   - Stratified sampling for classification
   - Save processed data to CSV

8. Export Cleaned Data
   - Save: diabetic_data_cleaned.csv
   - Save feature names for model reuse
```

**Output Files:**
- ✅ `diabetic_data_cleaned.csv` - Main cleaned dataset
- ✅ Feature lists for each model (selected_features_final.pkl, etc.)
- ✅ Scalers (scaler_readmission_final.pkl, etc.)

---

### **Notebook 2: Preprocessing.ipynb**
**Purpose:** Preprocessing for vital signs time-series data

**Dataset:** `human_vital_signs_dataset_2024.csv` (ICU monitoring data)

**Models Using This Preprocessing:**
1. ✅ **LSTM ICU Risk Prediction** - Time-series deep learning model

**Key Preprocessing Steps:**
```
1. Load Vital Signs Data
   - Patient ID, Heart Rate, Respiratory Rate, Temperature, SpO2, BP
   - Timestamp column for temporal ordering

2. Handle Missing Values
   - Forward fill for time-series continuity
   - Interpolation for gaps in vital readings

3. Feature Engineering
   - Derived_HRV: Heart Rate Variability
   - Derived_Pulse_Pressure: Systolic - Diastolic
   - Derived_MAP: Mean Arterial Pressure = (2*Diastolic + Systolic)/3
   - Derived_BMI: Weight / (Height)^2

4. Outlier Detection
   - Identify physiologically impossible values
   - Remove outliers using IQR method
   - Cap extreme values at clinical thresholds

5. Sequence Creation
   - Create time windows (e.g., 10 readings per sequence)
   - Sliding window approach for LSTM input
   - Ensure temporal ordering

6. Normalization
   - Min-Max scaling (0-1) for LSTM stability
   - Scale per feature independently

7. Risk Categorization
   - Label high-risk vs low-risk based on thresholds
   - One-hot encode risk categories

8. Export Cleaned Data
   - Save: cleaned_vital_dataset.csv
```

**Output Files:**
- ✅ `cleaned_vital_dataset.csv` - Time-series ready data

---

### **Notebook 3: Deep Learning Preprocessing (Sentiment)**
**Location:** `notebooks/deep learning/bert_sentiment/` folder

**Purpose:** Text preprocessing for patient feedback

**Dataset:** Patient feedback comments (CSV with text + sentiment labels)

**Models Using This Preprocessing:**
1. ✅ **BERT Sentiment Analysis** - Fine-tuned DistilBERT model

**Key Preprocessing Steps:**
```
1. Load Feedback Data
   - Text column: patient comments
   - Sentiment column: 0 (Negative), 1 (Positive)

2. Text Cleaning
   - Remove special characters and punctuation
   - Convert to lowercase
   - Remove extra whitespaces
   - Remove URLs and email addresses

3. Text Normalization
   - Expand contractions ("don't" → "do not")
   - Remove stopwords (optional for BERT)
   - Lemmatization (reduce words to root form)

4. Label Encoding
   - Ensure binary labels: 0 = Negative, 1 = Positive
   - Check class distribution (balance)

5. Train-Test Split
   - 80/20 split stratified by sentiment
   - Ensure balanced classes in both sets

6. BERT Tokenization
   - Use DistilBERT tokenizer
   - Max length: 128 tokens
   - Add [CLS] and [SEP] special tokens
   - Create attention masks

7. Export Cleaned Data
   - Save: cleaned_feedback_dataset.csv
   - Save tokenized inputs for training
```

**Output Files:**
- ✅ `cleaned_feedback_dataset.csv` - Text + labels
- ✅ `checkpoint-135/` - Trained BERT model

---

### **Notebook 4: Image Preprocessing (X-Ray)**
**Location:** `notebooks/deep learning/` (CNN training notebook)

**Purpose:** Image preprocessing for pneumonia detection

**Dataset:** `pneumoniamnist_balanced.npz` (8,136 chest X-ray images)

**Models Using This Preprocessing:**
1. ✅ **CNN Pneumonia Detection** - ResNet50 model

**Key Preprocessing Steps:**
```
1. Load X-Ray Dataset
   - Load .npz file (NumPy compressed format)
   - Extract: train_images, val_images, test_images
   - Extract: train_labels, val_labels, test_labels

2. Image Normalization
   - Scale pixel values: 0-255 → 0-1 (divide by 255)
   - Convert to float32 for GPU efficiency

3. Image Resizing
   - Original: 28×28 grayscale
   - Resize for ResNet50 input requirements
   - Maintain aspect ratio

4. Data Augmentation (Training only)
   - Random rotation (±15 degrees)
   - Random horizontal flip
   - Random zoom (90-110%)
   - Random brightness adjustment

5. Label Encoding
   - 0 = Normal chest X-ray
   - 1 = Pneumonia detected
   - One-hot encoding for Keras

6. Channel Conversion
   - Grayscale (1 channel) → RGB (3 channels)
   - Replicate grayscale across all channels

7. Dataset Balancing
   - Already balanced (47.5% Normal, 52.5% Pneumonia)
   - No SMOTE needed
```

**Output Files:**
- ✅ `pneumoniamnist_balanced.npz` - Preprocessed images
- ✅ Trained CNN model weights

---

## 📊 Preprocessing Summary Table

| Preprocessing Notebook | Dataset | Models | Key Techniques |
|------------------------|---------|--------|----------------|
| `preprocessing_version_1.ipynb` | Diabetic Data (101K rows) | Readmission, LOS, Clustering, Association | Encoding, Scaling, Imputation, Feature Engineering |
| `Preprocessing.ipynb` | Vital Signs (Time-series) | LSTM ICU Risk | Sequence Creation, Derived Features, Normalization |
| BERT Preprocessing | Patient Feedback (Text) | Sentiment Analysis | Text Cleaning, Tokenization, BERT Encoding |
| CNN Preprocessing | Chest X-Rays (Images) | Pneumonia Detection | Normalization, Augmentation, Resizing |

---

## 🎯 How to Explain to Evaluator

### **Opening Statement:**
> "Our project uses **4 different preprocessing pipelines** tailored to each data type: tabular, time-series, text, and images. Let me walk you through how preprocessing_version_1.ipynb handles all tabular models."

### **Step-by-Step Explanation:**

#### **1. Show preprocessing_version_1.ipynb:**
> "This single notebook preprocesses the diabetic dataset for 4 different models:
> - **Readmission prediction** needs binary labels and scaled features
> - **Length of Stay** needs same features but continuous target
> - **Clustering** needs normalized features (all 0-1 range)
> - **Association Rules** needs categorical binning (age groups, medication levels)
> 
> We handle all 4 use cases in one notebook because they share the same base dataset. The preprocessing steps are reusable, with minor variations saved as different output files."

#### **2. Highlight Modular Structure:**
> "Notice how we split preprocessing by data type:
> - **Tabular data** → preprocessing_version_1.ipynb (diabetic records)
> - **Time-series data** → Preprocessing.ipynb (vital signs)
> - **Text data** → BERT notebook (patient feedback)
> - **Image data** → CNN notebook (chest X-rays)
> 
> This modular approach makes it easy to maintain and update each pipeline independently."

#### **3. Show Output Files:**
> "Each preprocessing notebook produces cleaned datasets:
> - `diabetic_data_cleaned.csv` feeds into 4 models
> - `cleaned_vital_dataset.csv` feeds into LSTM
> - `cleaned_feedback_dataset.csv` feeds into BERT
> - `pneumoniamnist_balanced.npz` feeds into CNN
> 
> The dashboard loads these cleaned files, not the raw data, ensuring consistency."

#### **4. Demonstrate Reusability:**
> "For example, in preprocessing_version_1.ipynb:
> - We create `readmitted_binary` for classification
> - We keep `time_in_hospital` continuous for regression
> - We normalize all features for clustering
> - We bin features into categories for association rules
> 
> One preprocessing notebook, multiple outputs, zero redundancy."

---

## 🔍 Common Evaluator Questions & Answers

### **Q: Why use one notebook for multiple models?**
> "Because all 4 models (Readmission, LOS, Clustering, Association) use the **same diabetic dataset**. Instead of repeating preprocessing 4 times, we do it once efficiently. The only differences are:
> - Target variable handling (binary vs continuous)
> - Feature scaling method (StandardScaler vs MinMaxScaler)
> - Feature selection (different models need different feature subsets)"

### **Q: How do you prevent data leakage?**
> "We apply preprocessing steps in the correct order:
> 1. **Split data first** (train/test) before any transformations
> 2. **Fit scalers on training data only**, then transform test data
> 3. **Save fitted scalers** to use on new predictions
> 4. Never use test data to calculate statistics like mean, std, etc."

### **Q: Why different preprocessing for each data type?**
> "Each data type has unique requirements:
> - **Tabular:** Needs encoding, scaling, imputation
> - **Time-series:** Needs sequence creation, temporal ordering
> - **Text:** Needs tokenization, BERT embeddings
> - **Images:** Needs normalization, augmentation
> 
> A single preprocessing approach wouldn't work across all types."

### **Q: How do you handle class imbalance?**
> "We check balance in each preprocessing notebook:
> - **Diabetic data:** 11% readmission rate (imbalanced) → Use XGBoost `scale_pos_weight`
> - **X-ray data:** 52.5% pneumonia (balanced) → No special handling needed
> - **Sentiment data:** Check ratio → Apply class weights if needed
> - **Vital signs:** Risk categories balanced → No SMOTE required"

---

## 📂 File Organization

```
HealthCare System/
├── Data/
│   ├── Raw/
│   │   ├── diabetic_data.csv
│   │   ├── human_vital_signs_dataset_2024.csv
│   │   └── pneumoniamnist.npz
│   └── Processed/
│       ├── diabetic_data_cleaned.csv         ← From preprocessing_version_1.ipynb
│       ├── cleaned_vital_dataset.csv         ← From Preprocessing.ipynb
│       ├── cleaned_feedback_dataset.csv      ← From BERT preprocessing
│       └── pneumoniamnist_balanced.npz       ← From CNN preprocessing
│
├── notebooks/
│   ├── preprocessing_version_1.ipynb         ← Tabular (4 models)
│   ├── Preprocessing.ipynb                   ← Time-series (LSTM)
│   ├── classification_version_1.ipynb        ← Uses diabetic_data_cleaned.csv
│   ├── regression_version_1.ipynb            ← Uses diabetic_data_cleaned.csv
│   ├── clustering_version_1.ipynb            ← Uses diabetic_data_cleaned.csv
│   ├── Association_version_1.ipynb           ← Uses diabetic_data_cleaned.csv
│   └── deep learning/
│       ├── bert_sentiment/                   ← Text preprocessing + training
│       └── cnn_pneumonia/                    ← Image preprocessing + training
│
└── Trained_Models/
    ├── Readmission/                          ← From preprocessing_version_1.ipynb
    ├── LOS_Prediction/                       ← From preprocessing_version_1.ipynb
    ├── Clustering/                           ← From preprocessing_version_1.ipynb
    ├── Sentiment/                            ← From BERT preprocessing
    └── CNN/                                  ← From CNN preprocessing
```

---

## ✅ Evaluation Checklist

Before presenting to evaluator, ensure:

- [ ] All preprocessing notebooks run without errors
- [ ] Output files exist in `Data/Processed/` folder
- [ ] Can explain why preprocessing_version_1.ipynb handles 4 models
- [ ] Can demonstrate modular preprocessing by data type
- [ ] Can show how cleaned data flows into models
- [ ] Can explain data leakage prevention strategies
- [ ] Can justify preprocessing choices for each data type

---

## 🚀 Quick Demo Flow

**1. Show File Structure** (30 seconds)
   - Point to `Data/Processed/` → Show 4 cleaned datasets
   - Point to `notebooks/` → Show 2 main preprocessing notebooks

**2. Open preprocessing_version_1.ipynb** (2 minutes)
   - Scroll through sections: Loading → Cleaning → Encoding → Scaling → Export
   - Highlight: "One dataset, four models, same preprocessing base"
   - Show final cell: Saves `diabetic_data_cleaned.csv`

**3. Open Dashboard App.py** (1 minute)
   - Show: `load_diabetic_data()` loads the cleaned CSV
   - Point out: All 4 models (Readmission, LOS, Clustering, Association) use this same file

**4. Demonstrate Modular Approach** (1 minute)
   - Show: `Preprocessing.ipynb` for vital signs (completely separate)
   - Show: BERT preprocessing folder (text is different from tabular)
   - Emphasize: Each data type has its own optimized pipeline

**5. Conclusion** (30 seconds)
   - "By organizing preprocessing by data type instead of by model, we avoid redundancy and ensure consistency across all 4 diabetic models while maintaining flexibility for time-series, text, and image data."

---

**Total Demo Time: ~5 minutes**

This structured approach demonstrates thoughtful engineering and clear separation of concerns! 🎯
