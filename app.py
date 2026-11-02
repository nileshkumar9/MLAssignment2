# app.py - Part 1: Imports and Core Artifact Loading
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# Set page title and layout structure
st.set_page_config(page_title="AIML Machine Learning Evaluator", layout="wide")

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("### BITS Pilani M.Tech (AIML) - Assignment 2")
st.write("Upload your test dataset to evaluate our pre-trained classification models in real time.")

# Helper function to safely load saved modeling assets
@st.cache_resource
def load_assignment_artifacts():
    try:
        scaler = joblib.load("model/scaler.joblib")
        models = {
            "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
            "Decision Tree": joblib.load("model/decision_tree.joblib"),
            "KNN": joblib.load("model/knn.joblib"),
            "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
            "Support Vector Machine": joblib.load("model/support_vector_machine.joblib"),
        }
        return scaler, models
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts. Check if they exist in your 'model/' folder. Error: {e}")
        return None, None

scaler, models = load_assignment_artifacts()



# app.py - Part 2: Interactive UI, Prediction, and Evaluation Layout

if scaler and models:
    # 1. Sidebar Control Panel for File Upload and Model Selection Dropdown
    st.sidebar.header("📁 Control Panel")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test CSV File", 
        type=["csv"], 
        help="Upload the 'test_data.csv' file generated during your modeling pipeline."
    )
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model for Evaluation",
        list(models.keys())
    )

    # 2. Main Interface Logic
    if uploaded_file is not None:
        # Load the uploaded file
        test_df = pd.read_csv(uploaded_file)
        
        # Verify the target column exists
        if "Churn" not in test_df.columns:
            st.error("❌ The uploaded CSV must contain the target column labeled exactly 'Churn'.")
        else:
            st.success("✅ Test dataset uploaded and read successfully!")
            
            # Extract features and target values
            y_test_app = test_df["Churn"]
            X_test_app = test_df.drop(columns=["Churn"])
            
            # Scale the continuous features using our saved training scaler parameters
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            X_test_scaled_app = X_test_app.copy()
            X_test_scaled_app[numerical_cols] = scaler.transform(X_test_app[numerical_cols])
            
            # Retrieve the specific selected model object
            active_model = models[selected_model_name]
            
            # Generate predictions and probabilities without retraining
            y_pred_app = active_model.predict(X_test_scaled_app)
            
            if hasattr(active_model, "predict_proba"):
                y_prob_app = active_model.predict_proba(X_test_scaled_app)[:, 1]
            else:
                y_prob_app = active_model.decision_function(X_test_scaled_app)
                
            # 3. Compute the 6 Mandatory Assignment Metrics
            acc = accuracy_score(y_test_app, y_pred_app)
            auc = roc_auc_score(y_test_app, y_prob_app)
            prec = precision_score(y_test_app, y_pred_app)
            rec = recall_score(y_test_app, y_pred_app)
            f1 = f1_score(y_test_app, y_pred_app)
            mcc = matthews_corrcoef(y_test_app, y_pred_app)
            
            # 4. Display Metrics Panel Grid Layout
            st.markdown(f"## 🎯 Performance Metrics for: **{selected_model_name}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Accuracy", value=f"{acc:.4f}")
                st.metric(label="Recall (Sensitivity)", value=f"{rec:.4f}")
            with col2:
                st.metric(label="AUC Score", value=f"{auc:.4f}")
                st.metric(label="F1-Score", value=f"{f1:.4f}")
            with col3:
                st.metric(label="Precision", value=f"{prec:.4f}")
                st.metric(label="MCC Score", value=f"{mcc:.4f}")
                
            # 5. Build and Display the Confusion Matrix
            st.markdown("### 📊 Confusion Matrix Breakdown")
            cm = confusion_matrix(y_test_app, y_pred_app)
            
            # Create a clean display grid layout for the matrix counts
            cm_df = pd.DataFrame(
                cm, 
                index=["Actual No Churn (0)", "Actual Churn (1)"], 
                columns=["Predicted No Churn (0)", "Predicted Churn (1)"]
            )
            st.table(cm_df)
            
            # Detailed row item counts summary for additional grader clarity
            st.info(
                f"**Summary Insights:** \n"
                f"* Correctly identified non-churners (True Negatives): **{cm[0][0]}** \n"
                f"* Correctly identified actual churners (True Positives): **{cm[1][1]}** \n"
                f"* Non-churners flagged as churning (False Positives): **{cm[0][1]}** \n"
                f"* Churners missed completely by the model (False Negatives): **{cm[1][0]}**"
            )
            
    else:
        # Prompt state displayed before user uploads a file
        st.info("💡 Please upload your 'test_data.csv' file using the sidebar panel to see model outputs.")
