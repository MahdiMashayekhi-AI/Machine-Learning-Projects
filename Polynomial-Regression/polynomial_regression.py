# -*- coding: utf-8 -*-
"""Polynomial_Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IQ42Dh9srrP_GY4iy0jjMIZe3_i27QeH

# Polynomial Regression

## Import libraries
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""## Import dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
df = pd.DataFrame(dataset)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1]

df.head()

pd.DataFrame(X).head()

"""## Linear Regression Model"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

"""## Polynomial Regression Model"""

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

"""## Visualing the linear model"""

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Visualling the polynomial model"""

plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Predicting with linear model"""

regressor.predict([[6.5]])

"""## Predicting with polynomial model"""

model.predict(poly.fit_transform([[4.5]]))