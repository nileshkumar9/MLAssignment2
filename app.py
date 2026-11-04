# app.py - Part 1: Imports and Core Artifact Loading
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

APP_DIR = Path(__file__).resolve().parent
NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Set page title and layout structure
st.set_page_config(page_title="AIML Machine Learning Evaluator", layout="wide")

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("### Student ID : 2026AK05126 - ML Assignment 2")
st.write("Upload your test dataset to evaluate our pre-trained classification models in real time.")

with st.expander("About the Dataset used"):
    st.markdown(
        """
        **Dataset source Used:** [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn)

        This project uses the **IBM Telco Customer Churn dataset**, which contains
        7,043 telecommunications customer records. The models predict whether a
        customer will churn (`1`) or remain with the company (`0`) using demographic,
        account, service, contract, and billing information.

        - **Input features:** 19 customer attributes, including tenure,
          contract type, internet services, payment method, and monthly charges.
        - **Target:** `Churn`, encoded as `Yes`/`No` or `1`/`0`.
        - **Preprocessing:** categorical features are one-hot encoded, while
          `tenure`, `MonthlyCharges`, and `TotalCharges` are standardized.
        - **Dataset use:** the saved models were trained before deployment. This
          app only preprocesses uploaded test data, generates predictions, and
          compares them with the provided target values; it does not retrain the
          models.



        You can upload either the raw customer format or the preprocessed
        `test_data.csv` format using the sidebar.
        """
    )

with st.expander("Download Sample Test Files"):
    st.write(
        "Choose either sample format below. The app automatically preprocesses the "
        "raw format before sending it to the saved models."
    )

    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        with st.container(border=True):
            st.markdown("**Preprocessed model-input format**")
            st.caption(
                "One-hot encoded feature columns matching the saved model input "
                "schema, with the `Churn` target column."
            )
            encoded_sample_path = APP_DIR / "test_data.csv"
            if encoded_sample_path.exists():
                st.download_button(
                    "Download encoded sample",
                    data=encoded_sample_path.read_bytes(),
                    file_name=encoded_sample_path.name,
                    mime="text/csv",
                )
            else:
                st.warning("Encoded sample file is not available in the repository.")

    with sample_col2:
        with st.container(border=True):
            st.markdown("**Raw, user-friendly format**")
            st.caption(
                "Categorical values such as `Yes`, `No`, `DSL`, and `Two year`, "
                "with the `Churn` target column."
            )
            raw_sample_path = APP_DIR / "telecom_churn_test_sample.csv"
            if raw_sample_path.exists():
                st.download_button(
                    "Download raw sample",
                    data=raw_sample_path.read_bytes(),
                    file_name=raw_sample_path.name,
                    mime="text/csv",
                )
            else:
                st.warning("Raw sample file is not available in the repository.")

# Helper function to safely load saved modeling assets
@st.cache_resource
def load_assignment_artifacts():
    try:
        model_dir = APP_DIR / "model"
        scaler = joblib.load(model_dir / "scaler.joblib")
        models = {
            "Logistic Regression": joblib.load(model_dir / "logistic_regression.joblib"),
            "Decision Tree": joblib.load(model_dir / "decision_tree.joblib"),
            "KNN": joblib.load(model_dir / "knn.joblib"),
            "Naive Bayes": joblib.load(model_dir / "naive_bayes.joblib"),
            "Support Vector Machine": joblib.load(model_dir / "support_vector_machine.joblib"),
        }
        return scaler, models
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts. Check if they exist in your 'model/' folder. Error: {e}")
        return None, None

scaler, models = load_assignment_artifacts()

def prepare_uploaded_data(test_df, active_model):
    """Convert raw or encoded upload data into the model's training schema."""
    if "Churn" not in test_df.columns:
        raise ValueError("The uploaded CSV must contain a target column named 'Churn'.")

    y_test = test_df["Churn"].replace({"Yes": 1, "No": 0})
    if y_test.isna().any() or not y_test.isin([0, 1]).all():
        raise ValueError("The 'Churn' column must contain only Yes/No or 0/1 values.")

    X_test = test_df.drop(columns=["Churn"]).copy()
    expected_columns = list(getattr(active_model, "feature_names_in_", []))

    if set(RAW_FEATURE_COLUMNS).issubset(X_test.columns):
        X_test["TotalCharges"] = pd.to_numeric(X_test["TotalCharges"], errors="coerce").fillna(0.0)
        X_test = pd.get_dummies(
            X_test[RAW_FEATURE_COLUMNS],
            columns=[column for column in RAW_FEATURE_COLUMNS if column not in NUMERICAL_COLUMNS],
            drop_first=True,
        )
    elif not expected_columns or set(expected_columns) != set(X_test.columns):
        missing = sorted(set(expected_columns) - set(X_test.columns))
        extra = sorted(set(X_test.columns) - set(expected_columns))
        raise ValueError(
            "The uploaded CSV has an unsupported feature schema. "
            f"Missing columns: {missing or 'none'}; extra columns: {extra or 'none'}."
        )

    if expected_columns:
        X_test = X_test.reindex(columns=expected_columns, fill_value=0)

    missing_numeric = [column for column in NUMERICAL_COLUMNS if column not in X_test]
    if missing_numeric:
        raise ValueError(f"Missing required numerical columns: {missing_numeric}.")

    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    return X_test, y_test.astype(int)



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
        
        try:
            active_model = models[selected_model_name]
            X_test_scaled_app, y_test_app = prepare_uploaded_data(test_df, active_model)
            st.success("✅ Test dataset uploaded and read successfully!")

            # Generate predictions and probabilities without retraining
            y_pred_app = active_model.predict(X_test_scaled_app)
            
            if hasattr(active_model, "predict_proba"):
                y_prob_app = active_model.predict_proba(X_test_scaled_app)[:, 1]
            else:
                y_prob_app = active_model.decision_function(X_test_scaled_app)
                
            # 3. Compute the 6 Mandatory Assignment Metrics
            acc = accuracy_score(y_test_app, y_pred_app)
            auc = roc_auc_score(y_test_app, y_prob_app)
            prec = precision_score(y_test_app, y_pred_app, zero_division=0)
            rec = recall_score(y_test_app, y_pred_app, zero_division=0)
            f1 = f1_score(y_test_app, y_pred_app, zero_division=0)
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
        except ValueError as error:
            st.error(f"❌ {error}")
        except (KeyError, TypeError) as error:
            st.error(f"❌ Could not process the uploaded CSV: {error}")
    else:
        # Prompt state displayed before user uploads a file
        st.info("💡 Please upload your 'test_data.csv' file using the sidebar panel to see model outputs.")
