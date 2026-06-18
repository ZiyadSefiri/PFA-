#!/usr/bin/env python
# coding: utf-8

# # Training Notebook - Logistic Regression Pipeline
# 
# This notebook sets up a scikit-learn pipeline mirroring the EDA preprocessing workflow. It trains a `LogisticRegression` model, evaluates its capabilities, and logs metrics/models to MLflow.

# In[ ]:


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from utility.main import read_data

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def load_and_initial_clean():
    df = read_data()
    # Replace ? with nan
    df = df.replace('?', np.nan)
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop columns with high missingness
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop rows missing critical values
    df = df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3', 'gender'])

    # Drop all null columns
    all_null_cols = df.columns[df.isnull().all()]
    df = df.drop(columns=all_null_cols, errors='ignore')

    # Remove Unknown/Invalid gender
    df = df[df['gender'] != 'Unknown/Invalid']

    # Create target variable (1 if readmitted <30 days)
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Drop IDs and original target to prevent data leakage
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'], errors='ignore')

    return df

df = load_and_initial_clean()
print(f"Data shape after initial cleaning: {df.shape}")


# In[ ]:


class CustomMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. Age Mapping
        age_map = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, 
            '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, 
            '[80-90)': 8, '[90-100)': 9
        }
        if 'age' in X_copy.columns:
            X_copy['age'] = X_copy['age'].replace(age_map)

        # 2. Binary Mappings
        if 'change' in X_copy.columns:
            X_copy['change'] = X_copy['change'].replace({'No': 0, 'Ch': 1}).astype(float)
        if 'diabetesMed' in X_copy.columns:
            X_copy['diabetesMed'] = X_copy['diabetesMed'].replace({'No': 0, 'Yes': 1}).astype(float)
        if 'gender' in X_copy.columns:
            X_copy['gender'] = X_copy['gender'].replace({'Male': 0, 'Female': 1}).astype(float)

        # 3. Medication mapping (No=0, Steady/Up/Down=1)
        medication_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]

        for col in medication_cols:
            if col in X_copy.columns:
                X_copy[col] = (X_copy[col] != 'No').astype(float)

        # Constant columns from EDA uniqueness test won't harm the pipeline;
        # but we drop them here if they're present
        constant_cols = ['citoglipton', 'metformin-rosiglitazone', 'examide']
        X_copy = X_copy.drop(columns=constant_cols, errors='ignore')

        return X_copy


# In[ ]:


# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Grouping columns for specific transformations
num_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]
diag_cols = ['diag_1', 'diag_2', 'diag_3']
cat_cols = ['race', 'max_glu_serum', 'A1Cresult']

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Target Encoder handles diagnoses columns
# Note: smoothing mitigates data-leakage / overfitting
target_encoder_transformer = TargetEncoder(smooth='auto')

col_transformer = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('diag_target_enc', target_encoder_transformer, diag_cols)
    ],
    remainder='passthrough'  # Keep transformed age, gender, medication from CustomMapper
)

preprocessor = Pipeline(steps=[
    ('custom_mapper', CustomMapper()),
    ('col_transform', col_transformer)
])

# Preview pipeline components
preprocessor


# In[ ]:


# Setup MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetic_Readmission")

with mlflow.start_run(run_name="Logistic_Regression_Pipeline"):
    # Define Model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # Full classification pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("Training Pipeline Configured. Fitting model...")
    full_pipeline.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Logging to MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_params(model.get_params())

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log the complete scikit-learn pipeline
    mlflow.sklearn.log_model(full_pipeline, "model")

    print("Logging to MLflow complete!")

