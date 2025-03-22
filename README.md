This project implements a fraud detection system for mobile money transactions using machine learning. 
It features a Streamlit-based dashboard to visualize model performance, predictions, feature importance, and monitoring trends. 
The system uses XGBoost with SMOTE for handling imbalanced data, persists the trained model, and logs predictions for ongoing analysis.
Overview
This project builds a fraud detection model for mobile money transactions using engineered features like transaction amount, frequency, and location changes. It includes:

Training an XGBoost classifier with SMOTE to address class imbalance.
A Streamlit dashboard with pages for performance overview, predictions, feature importance, and monitoring.
Model persistence and prediction logging for deployment and tracking.
The dataset (mobile_money_features.csv) is assumed to contain transaction data with a Fraud_Label column (0 for legit, 1 for fraud).

Features
Model Training: Trains an XGBoost model with SMOTE if no saved model exists; otherwise, loads the persisted model.
Dashboard Pages:
Overview: Displays class distribution, classification report, and confusion matrix.
Predictions: Shows test set predictions with actual vs. predicted labels.
Feature Importance: Visualizes the importance of each feature in fraud detection.
Monitoring: Logs predictions over time and plots fraud detection trends.
Persistence: Saves the trained model as fraud_model.pkl and predictions as fraud_predictions.csv.
Logging: Appends predictions with timestamps to prediction_log.csv for monitoring.
Monitoring Strategy
Prediction Logging: Each run appends predictions to prediction_log.csv with timestamps.
Dashboard Monitoring: The "Monitoring" page shows the latest 10 predictions and a trend plot of fraud detections.
Metrics to Track:
Fraud recall: Are we catching frauds?
False positive rate: Are we flagging too many legit transactions?
Model drift: Compare performance on new data over time.
For real-time monitoring, integrate with a live data feed (e.g., API) in future iterations.

Future Improvements
Real-Time Data: Adapt the script to process live transactions via an API.
Model Tuning: Optimize XGBoost hyperparameters for better performance.
Alerts: Add notifications for high fraud detection rates in the monitoring page.
Scalability: Deploy with a database (e.g., SQLite) for persistent logging.
Explainability: Integrate SHAP or LIME to explain individual predictions.
Contributing
Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m "Add new feature").
Push to the branch (git push origin feature/new-feature).
Open a pull request.
