import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("P14-Part2-Regression/Section 8 - Polynomial Regression/Python/Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Linear regression     

linear_regression = LinearRegression()
linear_regression.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
print(X_poly)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
