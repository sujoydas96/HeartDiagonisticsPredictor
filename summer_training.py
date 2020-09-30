# -*- coding: utf-8 -*-
"""Summer Training

Importing Libraries
"""

import numpy as np
import pandas as pd

"""Importing the Dataset"""

dataset = pd.read_csv('Heart_Disease.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""Splitting to Test Set and Training Set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""Label Encoding and Dummy Trap Avoidance"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

Col_T = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = Col_T.fit_transform(X)

"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Fitting Support Vector Machine to Training Set"""

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train, y_train)

"""Predicting the Test Set Results"""

Y_Pred = classifier.predict(X_test)

"""Checking the Accuracy of the Results"""

from sklearn.metrics import accuracy_score
print("The accuracy attained using Support Vector Machine is " + str(accuracy_score(y_test, Y_Pred) * 100) + " percent")

"""Fitting Naive Bayes Classification to the Training Set"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""Predicting the Test Set results"""

Y_Pred = classifier.predict(X_test)

"""Checking the accuracy of the Results"""

from sklearn.metrics import accuracy_score
print("The accuracy attained using Naive Bayes Regression is " + str(accuracy_score(y_test, Y_Pred) * 100) + " percent")

"""Fitting K-Nearest Neighbours to the Training Set"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

"""Predicting the Test Set Results"""

y_pred = classifier.predict(X_test)

"""Checking the accuracy of the Results"""

from sklearn.metrics import accuracy_score
print("The accuracy attained using KNN Model is " + str(accuracy_score(y_test, Y_Pred) * 100) + " percent")

"""Fitting Decision Tree Regression to the Training Set"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

"""Predicting the Test Set Results"""

Y_Pred = regressor.predict(X_test)

"""Checking the accuracy of the Results"""

from sklearn.metrics import accuracy_score
print("The accuracy attained using Decision Tree Regression is " + str(accuracy_score(y_test, Y_Pred) * 100) + " percent")

"""Fitting Logistic Regression"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

"""Predicting the Test Set Results"""

Y_Pred = classifier.predict(X_test)

"""Checking the accuracy of the Results"""

from sklearn.metrics import accuracy_score
print("The accuracy attained using Logistic Regression is " + str(accuracy_score(y_test, Y_Pred) * 100) + " percent")