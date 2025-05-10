import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv(r"C:\Users\kanna\Downloads\LoanApprovalPrediction (1).csv")

# Identify categorical columns
obj_cols = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical variables:", len(obj_cols), obj_cols)

# Identify numerical columns
num_cols = data.select_dtypes(include=['int', 'float']).columns.tolist()
print("Numerical variables:", len(num_cols), num_cols)

# Drop 'Loan_ID' and update categorical columns list
if 'Loan_ID' in data.columns:
    data.drop(['Loan_ID'], axis=1, inplace=True)
    obj_cols.remove('Loan_ID')

# Plot categorical feature distributions
plt.figure(figsize=(12, 6))
for index, col in enumerate(obj_cols, 1):
    y = data[col].value_counts()
    plt.subplot(3, 3, index)  
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
plt.tight_layout()
plt.show()

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
data[obj_cols] = imputer.fit_transform(data[obj_cols])
data[num_cols] = imputer.fit_transform(data[num_cols])

# Label Encoding for categorical variables
label_encoder = preprocessing.LabelEncoder()
for col in obj_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Heatmap to show correlation
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidth=2, annot=True)
plt.show()

# Categorical plot for Gender vs Married vs Loan Status
sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.show()

# Split data into features and target
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Train-test split (60-40 ratio)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
print("Training Data Shape:", X_train.shape, Y_train.shape)
print("Testing Data Shape:", X_test.shape, Y_test.shape)

# Feature scaling for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(criterion="entropy", random_state=7)
svc = SVC()
lc = LogisticRegression()

# Train and evaluate models on training data
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print(f"Accuracy on Training Data ({clf.__class__.__name__}): {100 * metrics.accuracy_score(Y_train, Y_pred):.2f}%")

# Evaluate models on test data
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print(f"Accuracy on Test Data ({clf.__class__.__name__}): {100 * metrics.accuracy_score(Y_test, Y_pred):.2f}%")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming you have already trained your Logistic Regression model (lc)
# and your scaler (scaler) from your previous code.

# Example New Loan Application Data (Replace with your actual data)
new_loan_data = pd.DataFrame({
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': [2],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [6000],
    'CoapplicantIncome': [1500],
    'LoanAmount': [150],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban']
})

# Preprocessing (same as training)
obj_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
num_cols = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in obj_cols:
    new_loan_data[col] = label_encoder.fit_transform(new_loan_data[col])

new_loan_scaled = scaler.transform(new_loan_data)

# Make Predictions
predicted_prob = lc.predict_proba(new_loan_scaled)[:, 1]
predicted_class = lc.predict(new_loan_scaled)

# Confidence Check
uncertainty_threshold = 0.05  # Adjust as needed

if abs(predicted_prob[0] - 0.5) < uncertainty_threshold:
    confidence_message = "⚠️ Uncertain prediction (Low confidence)."
elif predicted_prob[0] > 0.5:
    confidence_message = "✅ Predicted: Loan Approved (High confidence)"
else:
    confidence_message = "🔻 Predicted: Loan Rejected (High confidence)"

# Print Results
print(f"Predicted Probability of Loan Approval: {predicted_prob[0]:.2f}")
print(f"Predicted Class (0=Rejected, 1=Approved): {predicted_class[0]}")
print(confidence_message)
