import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("P14-Part2-Regression/Section 7 - Multiple Linear Regression/Python/50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# dummy vars
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])

ct = ColumnTransformer([("Country", OneHotEncoder(drop="first"), [3])], remainder="passthrough")

X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(mean_squared_error(y_test, y_pred))

#   Build the optimal model
#
# from statsmodels.regression.linear_model import OLS
#
# X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# X_opt = X.astype('float64')[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())


regressor = LinearRegression()
regressor.fit(X_train[:, 2].reshape(-1, 1), y_train)

y_pred = regressor.predict(X_test[:, 2].reshape(-1, 1))

print(mean_squared_error(y_test, y_pred))
