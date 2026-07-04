<p align="center">
  <img src="banner.png" alt="IPL Match Winner Predictor">
</p>

<p align="center">
<a href="https://ipl-match-prediction-model.streamlit.app"><img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-success?style=for-the-badge" alt="Live Demo"></a>
</p>

---

##  Project Overview

This project predicts the winner of an IPL match using a **Random Forest Classifier** trained on historical IPL match data (2008–2016).

Users can select:

- 🏏 Season
- 📍 City
- 👥 Team 1 & Team 2
- 🪙 Toss Winner
- 🎯 Toss Decision

The application then predicts:

- 🏆 Match Winner
- 📊 Winning Probability of both teams
- 📈 Model Information

The machine learning workflow was built using **Scikit-Learn Pipelines**, while the prediction interface is deployed as a **Streamlit web application**.

---

# 🌐 Live Demo

### 🚀 Try it here

https://ipl-match-prediction-model.streamlit.app

---

## 📷 Application Preview

![Application Screenshot](screenshot.png)

---

##  Features

- 🌐 Live deployed Streamlit application
- 🏏 IPL match winner prediction
- 📊 Winning probability visualization
- 📝 Prediction history
- ✅ Input validation
- 🌙 Modern dark-themed UI
- ⚙️ Scikit-Learn preprocessing pipeline
- 📈 Model information panel

---

## 📊 Dataset

- **Source:** IPL Matches Dataset
- **Matches:** 577
- **Usable Matches:** 567
- **Seasons:** 2008–2016

### Input Features

- Season
- City
- Team 1
- Team 2
- Toss Winner
- Toss Decision

### Target

- Match Winner

---

## 🤖 Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Selection
   │
   ▼
Train/Test Split
   │
   ▼
ColumnTransformer
   │
   ▼
OneHotEncoder
   │
   ▼
Random Forest Classifier
   │
   ▼
Prediction
```

---

## 📈 Model Performance

### Model Comparison

![Model Comparison](model_comparison.png)

**Final Model:** Random Forest Classifier

**Accuracy:** **58.77%**

Random Forest outperformed Logistic Regression, KNN and Decision Tree, making it the final model used for deployment.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Joblib

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Preprocessing | ColumnTransformer + OneHotEncoder |
| Train/Test Split | 80/20 |
| Training Samples | 453 |
| Testing Samples | 114 |
| Random State | 42 |

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Team win-rate feature engineering
- Head-to-head statistics
- Venue-wise performance
- Recent form analysis
- XGBoost / LightGBM comparison
- Explainability using SHAP

---

## 🙏 Acknowledgements

The **EDA, feature engineering, preprocessing pipeline, model training, evaluation, and machine learning workflow** were developed by me.

AI assistance was used to:

- Build the Streamlit interface
- Refactor and organize the project structure
- Improve documentation and code readability

---

## ⭐ If you found this project interesting, consider giving it a star!
