# Customer Churn Classification Dashboard

Machine Learning Assignment 2 for the BITS Pilani M.Tech (AIML) programme.

## a. Problem statement

Customer churn is the loss of customers who stop using a company's services. The
objective of this project is to predict whether a telecommunications customer will
churn (`1`) or remain with the company (`0`) using customer demographic,
subscription, service, contract, and billing information.

Five classification models are trained on the same dataset and compared using
Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient
(MCC). The trained models are then made available through an interactive Streamlit
application.

## b. Dataset description

This project uses the IBM Telco Customer Churn dataset, a binary classification
dataset containing 7,043 customer records. The original data contains 19 input
features and one target column, `Churn`.

Important input fields include:

- Customer information: `gender`, `SeniorCitizen`, `Partner`, and `Dependents`
- Account information: `tenure`, `Contract`, `PaperlessBilling`, and `PaymentMethod`
- Service information: phone, internet, security, backup, streaming, and technical
  support services
- Billing information: `MonthlyCharges` and `TotalCharges`

The target column is encoded as:

- `0`: customer did not churn
- `1`: customer churned

Categorical features were one-hot encoded using `pandas.get_dummies(...,
drop_first=True)`. The continuous features `tenure`, `MonthlyCharges`, and
`TotalCharges` were standardized using `StandardScaler`. The data was split into
training and test sets using an 80:20 stratified split with `random_state=42`.

The public dataset is available from the
[Telecom Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn/data).

## c. GitHub repository link

[https://github.com/nileshkumar9/MLAssignment2](https://github.com/nileshkumar9/MLAssignment2)

Repository contents:

```text
MLAssignment2/
├── app.py
├── IT_customer_churn.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
├── test_data.csv
└── model/
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── naive_bayes.joblib
    ├── scaler.joblib
    └── support_vector_machine.joblib
```

## Live Streamlit application

**Live app URL:** `Add the deployed Streamlit Community Cloud URL here`

The application allows the evaluator to:

1. Upload `test_data.csv`.
2. Select one of the five saved classification models.
3. View the six required evaluation metrics.
4. View the model's confusion matrix and prediction summary.

The application loads the saved model and scaler artifacts. It does not retrain
models on the uploaded test data. The app accepts either:

- The original, user-friendly CSV format with categorical values such as `Yes`,
  `No`, `DSL`, and `Two year`, plus the `Churn` column.
- The already preprocessed, one-hot encoded format in `test_data.csv`.

When a raw CSV is uploaded, the app applies the same one-hot encoding and feature
alignment used during model training before making predictions.

## d. Models and evaluation results

The following results were obtained on the held-out test data.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8062 | 0.8422 | 0.6593 | 0.5588 | 0.6049 | 0.4806 |
| Decision Tree | 0.8006 | 0.8219 | 0.6667 | 0.4973 | 0.5697 | 0.4515 |
| K-Nearest Neighbors | 0.7708 | 0.8033 | 0.5706 | 0.5508 | 0.5605 | 0.4057 |
| Gaussian Naive Bayes | 0.6558 | 0.8092 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Support Vector Machine | 0.7949 | 0.7948 | 0.6545 | 0.4813 | 0.5547 | 0.4339 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the best overall balance, with the highest Accuracy, AUC, F1 Score, and MCC. It is the strongest general-purpose model for this dataset. |
| Decision Tree | Produced the second-highest Accuracy and the highest Precision, but its lower Recall indicates that it missed more churners than Logistic Regression. |
| K-Nearest Neighbors | Produced moderate results, but its Accuracy, F1 Score, and MCC were lower than those of Logistic Regression and Decision Tree. |
| Gaussian Naive Bayes | Achieved the highest Recall and identified most churners, but its low Precision caused many non-churners to be incorrectly flagged as churners. |
| Support Vector Machine | Produced competitive Accuracy and Precision, but its Recall, F1 Score, and AUC were lower than the leading models. |

### Overall winner

**Logistic Regression** is the overall winner for this dataset because it achieved
the highest Accuracy (0.8062), AUC (0.8422), F1 Score (0.6049), and MCC (0.4806).
It provides the best overall balance between identifying churners and avoiding
false churn predictions.

## Running the application locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, upload `test_data.csv` from this repository, and
select a model from the sidebar.

## Screenshot

Add the required screenshot showing assignment execution in the BITS Virtual Lab
to the final PDF submission.
