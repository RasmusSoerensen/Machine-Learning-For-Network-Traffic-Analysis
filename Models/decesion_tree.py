import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Import data
data = pd.read_csv("Data\\processed_data.csv")

#Splitting the data into features and target
X = data.drop("label", axis=1)
y = data["label"]

#Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Creating the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

#Predicting the target attribute
y_pred = dt.predict(X_test)

#Evaluating the model
print(classification_report(y_test, y_pred))

#print the accuracy score
print("Accuracy: ", accuracy_score(y_test, y_pred))