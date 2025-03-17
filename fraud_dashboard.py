# Step 4: Evaluation and Dashboard Creation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the engineered dataset
df = pd.read_csv('mobile_money_features.csv')
print("Dataset loaded. Shape:", df.shape)

# Define features and target
features = [
    'Amount', 'Rolling_Count_1h', 'Time_Delta', 'Rapid_Transaction',
    'Avg_Amount', 'Max_Amount', 'Min_Amount', 'Location_Change',
    'Unique_Devices', 'Send_Money_Ratio', 'Hour_of_Day'
]
X = df[features]
y = df['Fraud_Label']

# Split into train and test sets (70% train, 30% test for more fraud cases)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class (fraud) in training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Debug: Check class distribution
print("\nClass distribution in y_train (original):", np.bincount(y_train))
print("Class distribution in y_train (SMOTE):", np.bincount(y_train_smote))
print("Class distribution in y_test:", np.bincount(y_test))

# Train XGBoost model with SMOTE data
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = model.predict(X_test)

# Save predictions for dashboard
df_test = X_test.copy()
df_test['Actual_Label'] = y_test
df_test['Predicted_Label'] = y_pred
df_test.to_csv('fraud_predictions.csv', index=False)

# --- Streamlit Dashboard ---
st.title("Mobile Money Fraud Detection Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Predictions", "Feature Importance"])

# Overview Page
if page == "Overview":
    st.header("Model Performance Overview")
    st.write("### Class Distribution")
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

# Footer
st.sidebar.write("Built with Streamlit by [Your Name]")

# Console output for verification
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nPredictions saved as 'fraud_predictions.csv'")