import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Load the engineered dataset
df = pd.read_csv('mobile_money_features.csv')
print("Dataset loaded. Shape:", df.shape)

# Verify Fraud_Label column
print("Columns in dataset:", df.columns.tolist())
print("Unique values in Fraud_Label:", df['Fraud_Label'].unique())
print("Value counts in Fraud_Label:\n", df['Fraud_Label'].value_counts())

# Simulate fraud cases if only one class exists (temporary fix for testing)
if len(df['Fraud_Label'].unique()) < 2:
    print("Warning: Only one class found in Fraud_Label. Simulating fraud cases for testing.")
    np.random.seed(42)
    df['Fraud_Label'] = np.where(np.random.rand(len(df)) < 0.05, 1, 0)  # 5% fraud cases
    print("New value counts in Fraud_Label after simulation:\n", df['Fraud_Label'].value_counts())

# Define features and target
features = [
    'Amount', 'Rolling_Count_1h', 'Time_Delta', 'Rapid_Transaction',
    'Avg_Amount', 'Max_Amount', 'Min_Amount', 'Location_Change',
    'Unique_Devices', 'Send_Money_Ratio', 'Hour_of_Day'
]
X = df[features]
y = df['Fraud_Label']

# Model and log file paths
MODEL_PATH = 'fraud_model.pkl'
LOG_PATH = 'prediction_log.csv'

# Function to train and save the model
def train_and_save_model(X, y):
    # Manual split: 70% train, 30% test ensuring both classes
    legit_df = df[df['Fraud_Label'] == 0]
    fraud_df = df[df['Fraud_Label'] == 1]
    
    # Adjust split sizes based on actual fraud count after simulation
    n_fraud = len(fraud_df)
    n_legit = len(legit_df)
    test_size_fraud = max(1, int(n_fraud * 0.3))  # At least 1 fraud in test
    test_size_legit = int(n_legit * 0.3)

    X_legit_train, X_legit_test = train_test_split(legit_df[features], test_size=test_size_legit, random_state=42)
    y_legit_train, y_legit_test = train_test_split(legit_df['Fraud_Label'], test_size=test_size_legit, random_state=42)
    X_fraud_train, X_fraud_test = train_test_split(fraud_df[features], test_size=test_size_fraud, random_state=42)
    y_fraud_train, y_fraud_test = train_test_split(fraud_df['Fraud_Label'], test_size=test_size_fraud, random_state=42)

    X_train = pd.concat([X_legit_train, X_fraud_train])
    y_train = pd.concat([y_legit_train, y_fraud_train])
    X_test = pd.concat([X_legit_test, X_fraud_test])
    y_test = pd.concat([y_legit_test, y_fraud_test])

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_smote, y_train_smote)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    return model, X_train, y_train, X_train_smote, y_train_smote, X_test, y_test

# Load model if exists, otherwise train and save
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # Reconstruct test set for consistency
    legit_df = df[df['Fraud_Label'] == 0]
    fraud_df = df[df['Fraud_Label'] == 1]
    n_fraud = len(fraud_df)
    n_legit = len(legit_df)
    test_size_fraud = max(1, int(n_fraud * 0.3))
    test_size_legit = int(n_legit * 0.3)
    
    X_legit_test = legit_df[features].sample(n=test_size_legit, random_state=42)
    y_legit_test = legit_df['Fraud_Label'].loc[X_legit_test.index]
    X_fraud_test = fraud_df[features].sample(n=test_size_fraud, random_state=42)
    y_fraud_test = fraud_df['Fraud_Label'].loc[X_fraud_test.index]
    
    X_test = pd.concat([X_legit_test, X_fraud_test])
    y_test = pd.concat([y_legit_test, y_fraud_test])
    print(f"Model loaded from {MODEL_PATH}")
else:
    model, X_train, y_train, X_train_smote, y_train_smote, X_test, y_test = train_and_save_model(X, y)

# Predict on test set
y_pred = model.predict(X_test)

# Log predictions
def log_predictions(X_test, y_test, y_pred):
    log_df = X_test.copy()
    log_df['Actual_Label'] = y_test
    log_df['Predicted_Label'] = y_pred
    log_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, index=False)
    else:
        log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    print(f"Predictions logged to {LOG_PATH}")

log_predictions(X_test, y_test, y_pred)

# Save predictions for dashboard
df_test = X_test.copy()
df_test['Actual_Label'] = y_test
df_test['Predicted_Label'] = y_pred
df_test.to_csv('fraud_predictions.csv', index=False)

# Debug: Check class distribution
print("\nClass distribution in y_test:", np.bincount(y_test))

# --- Streamlit Dashboard ---
st.title("Mobile Money Fraud Detection Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Predictions", "Feature Importance", "Monitoring"])

# Overview Page
if page == "Overview":
    st.header("Model Performance Overview")
    st.write("### Class Distribution")
    # Use original train data if model was trained, otherwise skip
    if 'y_train' in locals():
        st.write(f"Training (Original): {np.bincount(y_train)} (Legit, Fraud)")
        st.write(f"Training (SMOTE): {np.bincount(y_train_smote)} (Legit, Fraud)")
    st.write(f"Test: {np.bincount(y_test)} (Legit, Fraud)")

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Predictions Page
elif page == "Predictions":
    st.header("Transaction Predictions")
    st.write("Showing test set predictions (Legit = 0, Fraud = 1)")
    st.dataframe(df_test)

# Feature Importance Page
elif page == "Feature Importance":
    st.header("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.write(feature_importance)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance in Fraud Detection')
    st.pyplot(fig)

# Monitoring Page
elif page == "Monitoring":
    st.header("Model Monitoring")
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.write("### Prediction Log")
        st.dataframe(log_df.tail(10))  # Show last 10 entries

        st.write("### Fraud Detection Over Time")
        fraud_counts = log_df.groupby('Timestamp')['Predicted_Label'].sum()
        fig, ax = plt.subplots()
        fraud_counts.plot(kind='line', ax=ax)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Number of Frauds Detected')
        ax.set_title('Fraud Detection Trend')
        st.pyplot(fig)
    else:
        st.write("No prediction logs available yet.")

# Footer
st.sidebar.write("Built with Streamlit by GIDEON_MUTINDA")

# Console output for verification
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nPredictions saved as 'fraud_predictions.csv'")