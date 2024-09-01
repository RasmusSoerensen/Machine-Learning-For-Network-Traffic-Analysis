import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


data = pd.read_csv("Data\\processed_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier()
gbm = GradientBoostingClassifier()
knn = KNeighborsClassifier()
lor = LogisticRegression()
lr = LinearRegression()
nb = GaussianNB()
rf = RandomForestClassifier()
svm = SVC()

models = [dt, gbm, knn, lor, lr, nb, rf, svm]

accuracy = []
name = []


for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(model.score(X_test, y_test))
    name.append(str(model))

#create a dictionary
dictionary = dict(zip(name, accuracy))

#rank the models
sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

print("Ranking of models based on accuracy: ")
print(sorted_dict)

