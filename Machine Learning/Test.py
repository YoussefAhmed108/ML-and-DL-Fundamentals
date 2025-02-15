import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.linear_model import Ridge as skRidgeRegression
from sklearn.linear_model import Lasso as skLassoRegression
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as skDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier as skGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as skGradientBoostingRegressor

from Linear.LogisticRegression import LogisticRegression
from Linear.LinearRegression import LinearRegression
from Linear.RidgeRegression import RidgeRegression
from Linear.LassoRegression import LassoRegression
from Trees.DecisionTreeClassifier import DecisionTreeClassifier
from Trees.DecisionTreeRegressor import DecisionTreeRegressor
from ensemble.RandomForestClassifier import RandomForestClassifier
from ensemble.RandomForestRegressor import RandomForestRegressor
from decomposition.PCA import PCA
from ensemble.GradientBoostingClassifier import GradientBoostingClassifier
from ensemble.GradientBoostingRegressor import GradientBoostingRegressor

from sklearn.tree import export_text



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
# my_lasso = LassoRegression()
# my_lasso.fit(X_train_reg, y_train_reg)
# print(my_lasso.coef_)
# print(my_lasso.intercept_)
# print(my_lasso.score(X_test_reg, y_test_reg))
# sk_lasso = skLassoRegression()
# sk_lasso.fit(X_train_reg, y_train_reg)
# print(sk_lasso.coef_)
# print(sk_lasso.intercept_)
# print(sk_lasso.score(X_test_reg, y_test_reg))

################################################
# Decision Tree Classifier
################################################
# my_dtc = DecisionTreeClassifier()
# my_dtc.fit(X_train_clf, y_train_clf)
# print(my_dtc.score(X_test_clf, y_test_clf))
# my_dtc.print_tree(feature_names=classification_data.feature_names)
# sk_dtc = skDecisionTreeClassifier()
# sk_dtc.fit(X_train_clf, y_train_clf)
# print(sk_dtc.score(X_test_clf, y_test_clf))
# tree_text = export_text(sk_dtc, feature_names=classification_data.feature_names)
# print(tree_text)

################################################
# Decision Tree Regressor
################################################
# my_dtr = DecisionTreeRegressor()
# my_dtr.fit(X_train_reg, y_train_reg)
# print(my_dtr.score(X_test_reg, y_test_reg))
# my_dtr.print_tree(feature_names=regression_data.feature_names)
# sk_dtr = skDecisionTreeRegressor()
# sk_dtr.fit(X_train_reg, y_train_reg)
# print(sk_dtr.score(X_test_reg, y_test_reg))
# tree_text = export_text(sk_dtr, feature_names=regression_data.feature_names)
# print(tree_text)

###################################################
# Random Forest Classifier
###################################################
# my_rfc = RandomForestClassifier(n_estimators=100)
# my_rfc.fit(X_train_clf, y_train_clf)
# print(my_rfc.score(X_test_clf, y_test_clf))
# sk_rfc = skRandomForestClassifier(n_estimators=100)
# sk_rfc.fit(X_train_clf, y_train_clf)
# print(sk_rfc.score(X_test_clf, y_test_clf))

###################################################
# Random Forest Regressor
###################################################
# my_rfr = RandomForestRegressor(n_estimators=100 , max_depth=5)
# my_rfr.fit(X_train_reg, y_train_reg)
# print(my_rfr.score(X_test_reg, y_test_reg))
# sk_rfr = skRandomForestRegressor(n_estimators=100 , max_depth=5)
# sk_rfr.fit(X_train_reg, y_train_reg)
# print(sk_rfr.score(X_test_reg, y_test_reg))

###################################################
# Gradient Boosting Classifier
###################################################
# my_gbc = GradientBoostingClassifier(n_estimators=100)
# my_gbc.fit(X_train_clf, y_train_clf)
# print(my_gbc.score(X_test_clf, y_test_clf))
# sk_gbc = skGradientBoostingClassifier(n_estimators=100)
# sk_gbc.fit(X_train_clf, y_train_clf)
# print(sk_gbc.score(X_test_clf, y_test_clf))

###################################################
# Gradient Boosting Regressor
###################################################
my_gbr = GradientBoostingRegressor(n_estimators=100)
my_gbr.fit(X_train_reg, y_train_reg)
print(my_gbr.score(X_test_reg, y_test_reg))
sk_gbr = skGradientBoostingRegressor(n_estimators=100)
sk_gbr.fit(X_train_reg, y_train_reg)
print(sk_gbr.score(X_test_reg, y_test_reg))





print("Hello World")