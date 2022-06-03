# Importing dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# Data Collection and processing

data = pd.read_csv("diabetes.csv")
print(data.head())
print(data.describe())
print(data["Outcome"].value_counts())
print(data.groupby("Outcome").mean())

X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

# Data Standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = data["Outcome"]

# Split Training and Test Dataset

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)
print(X_train.shape)
print(Y_train.shape)

# Model Evaluation

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
print("Accuracy Score for Test Dataset: ", accuracy_score(Y_test, Y_predict))

# Predictive System

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

inputAsArray = np.asarray(input_data)
inputDataReshape = inputAsArray.reshape(1, -1)

stdInputData = scaler.transform(inputDataReshape)

prediction = classifier.predict(stdInputData)
if prediction == 0:
    print("Person is Non-Diabetic")
else:
    print("Person is Diabetic")
