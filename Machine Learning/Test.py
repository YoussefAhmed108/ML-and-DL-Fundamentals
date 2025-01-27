import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.linear_model import Ridge as skRidgeRegression
from sklearn.linear_model import Lasso as skLassoRegression

from Linear.LogisticRegression import LogisticRegression
from Linear.LinearRegression import LinearRegression
from Linear.RidgeRegression import RidgeRegression
from Linear.LassoRegression import LassoRegression



# Load the dataset
regression_data = datasets.load_diabetes()
classification_data = datasets.load_wine()
# Scale the data using StandardScaler
scaler = StandardScaler()
regression_data.data = scaler.fit_transform(regression_data.data)
classification_data.data = scaler.fit_transform(classification_data.data)
# Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(regression_data.data, regression_data.target, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(classification_data.data, classification_data.target, test_size=0.2, random_state=42)

#########
# Linear Regression
#########
# my_lr = LinearRegression()
# my_lr.fit(X_train, y_train)
# print(my_lr.coef_)
# print(my_lr.intercept_)
# print(my_lr.score(X_test, y_test))

# sk_lr = skLinearRegression()
# sk_lr.fit(X_train, y_train)
# print(sk_lr.coef_)
# print(sk_lr.intercept_)
# print(sk_lr.score(X_test, y_test))


#######################
# Logistic Regression
#######################
# my_logitr = LogisticRegression()
# my_logitr.fit(X_train_clf, y_train_clf)
# print(my_logitr.coef_)
# print(my_logitr.intercept_)
# print(my_logitr.score(X_test_clf, y_test_clf))
# sk_logitr = skLogisticRegression()
# sk_logitr.fit(X_train_clf, y_train_clf)
# print(sk_logitr.coef_)
# print(sk_logitr.intercept_)
# print(sk_logitr.score(X_test_clf, y_test_clf))

########################
# Ridge Regression
########################
# my_ridge = RidgeRegression()
# my_ridge.fit(X_train_reg, y_train_reg)
# print(my_ridge.coef_)
# print(my_ridge.intercept_)
# print(my_ridge.score(X_test_reg, y_test_reg))
# sk_ridge = skRidgeRegression()
# sk_ridge.fit(X_train_reg, y_train_reg)
# print(sk_ridge.coef_)
# print(sk_ridge.intercept_)
# print(sk_ridge.score(X_test_reg, y_test_reg))

#################################
# Lasso Regression
#################################
my_lasso = LassoRegression()
my_lasso.fit(X_train_reg, y_train_reg)
print(my_lasso.coef_)
print(my_lasso.intercept_)
print(my_lasso.score(X_test_reg, y_test_reg))
sk_lasso = skLassoRegression()
sk_lasso.fit(X_train_reg, y_train_reg)
print(sk_lasso.coef_)
print(sk_lasso.intercept_)
print(sk_lasso.score(X_test_reg, y_test_reg))


print("Hello World")