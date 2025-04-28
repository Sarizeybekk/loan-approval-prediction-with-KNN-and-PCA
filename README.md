# Loan Approval Prediction Web Application

This project is a complete machine learning pipeline and a Flask web application designed to predict loan approvals based on user input.

## Overview

- Machine Learning model trained using KNN with PCA and hyperparameter tuning (GridSearchCV).
- Model evaluation using confusion matrix, classification report, and ROC curve.
- Trained model is saved as `loan_approval_model.pkl`.
- Flask web application provides a simple interface for predictions.

---

## Dataset Features Used for Training

The model is trained on the following 11 features:

- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

> Note: `loan_id` column is excluded because it is an identifier and not predictive.

---

## Machine Learning Pipeline

1. **Data Preprocessing**
   - Label encoding for categorical features (`education`, `self_employed`, `loan_status`).
   - Dropping irrelevant columns (`loan_id`).

2. **Feature Scaling**
   - StandardScaler is applied to standardize features.

3. **Dimensionality Reduction**
   - PCA (Principal Component Analysis) reduces dimensions to improve generalization.

4. **Model Training**
   - K-Nearest Neighbors (KNN) classifier.
   - GridSearchCV for hyperparameter tuning.

5. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - Accuracy Score
   - ROC Curve Plot
   - 2D PCA Visualization

6. **Model Saving**
   - The final trained model is saved as `loan_approval_model.pkl`.

---

## Requirements

- Python 3.x
- Flask
- pandas
- scikit-learn
- matplotlib
- seaborn

Install all required packages using:

```bash
pip install -r requirements.txt
```
## Dataset Features Used for Training

The model is trained on the following 11 features:

- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

   # Proje Kurulum ve Çalıştırma Komutları

```bash
# 1. Projeyi klonla
git clone https://github.com/Sarizeybekk/loan-approval-prediction-with-KNN-and-PCA.git
cd loan-approval-prediction-with-KNN-and-PCA

# 2. Sanal ortam oluştur 
python -m venv venv
source venv/bin/activate  # Mac/Linux

# 3. Gerekli paketleri yükle
pip install -r requirements.txt

# 4. Flask uygulamasını başlat
python app.py

```
![image](https://github.com/user-attachments/assets/a7d00c12-f220-48f5-b66f-6e9a183c2c49)
