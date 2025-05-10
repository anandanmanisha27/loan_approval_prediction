readme_content = """
# 🏦 Loan Approval Prediction

This project aims to predict whether a loan application will be approved or rejected using machine learning algorithms. The dataset includes personal and financial information of applicants and the loan approval status.

## 📁 Dataset

- Source: Kaggle/Other
- Format: CSV
- Key Features:
  - Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
  - Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

## 🧹 Data Preprocessing

- Dropped `Loan_ID` as it's an identifier.
- Imputed missing values using the **most frequent strategy**.
- Applied **Label Encoding** for categorical variables.
- Scaled numerical features using **StandardScaler**.

## 📊 Exploratory Data Analysis

- Bar plots for categorical feature distributions.
- Heatmap of feature correlation.
- Visual relationship between variables like Gender, Married, and Loan Status.

## 🧠 Machine Learning Models Used

The following models were trained and evaluated:
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

### 🔎 Evaluation Metrics
- Accuracy on training and test datasets

## 🚀 Predicting on New Loan Applications

An example prediction is demonstrated for a new applicant. The model outputs:
- Predicted Class (0 = Rejected, 1 = Approved)
- Predicted Probability of Approval
- Confidence message based on how close the probability is to 0.5
