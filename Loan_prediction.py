import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load data
data = pd.read_csv("C:/Users/kanna/Downloads/LoanApprovalPrediction.csv")

# Display first 5 rows
print(data.head(5))

# Drop Loan_ID column
data.drop(['Loan_ID'], axis=1, inplace=True)

# Count and display number of categorical variables
obj = (data.dtypes == 'object')
print("Categorical variables:", len(list(obj[obj].index)))

# Plot bar plots for categorical variables
object_cols = list(obj[obj].index)
plt.figure(figsize=(18, 36))
index = 1

for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.show()

# Label encode categorical variables
label_encoder = preprocessing.LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Confirm all categorical variables are encoded
obj = (data.dtypes == 'object')
print("Categorical variables:", len(list(obj[obj].index)))

# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Plot categorical plot for Gender and Married status
sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.show()

# Fill missing values with mean
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# Check for any remaining missing values
print(data.isna().sum())

# Split data into features and target variable
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
print(X.shape, Y.shape)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Define classifiers
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
svc = SVC()
lc = LogisticRegression()

# Train and evaluate classifiers on training set
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score on training set of ", clf.__class__.__name__, "=", 100 * metrics.accuracy_score(Y_train, Y_pred))

# Train and evaluate classifiers on testing set
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score on testing set of ", clf.__class__.__name__, "=", 100 * metrics.accuracy_score(Y_test, Y_pred))
