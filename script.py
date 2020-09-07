import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./P14-Part2-Regression/Section 6 - Simple Linear Regression/Python/Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
#
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.show()


plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.show()
