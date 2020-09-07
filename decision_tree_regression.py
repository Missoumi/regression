import pandas as pd

dataset = pd.read_csv("./P14-Part2-Regression/Section 10 - Decision Tree Regression/Python/Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

from matplotlib import pyplot as plt
y_pred = regressor.predict([[6.5]])

print(y_pred)

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color="blue")
plt.show() 

