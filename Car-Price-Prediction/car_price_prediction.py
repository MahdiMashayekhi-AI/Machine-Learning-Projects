# -*- coding: utf-8 -*-
"""Car_Price_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16X2ZQ7Xhn6ooz3KERKdjAOJRUqYoQ7LE

# Car Price Prediction

A Chinese automobile company Teclov_chinese aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market.
"""

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('CarPrice_Assignment.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
dataset.head()

X.head()

y.head()

list(dataset.columns)

dataset.describe()

dataset.info()

dataset.shape

X.drop(['car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel'], axis=1, inplace=True)
X.head()

list(X.columns)

y.head()

"""## Preprocessing"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.iloc[:, 0] = le.fit_transform(X.iloc[:, 0])
X['enginetype'] = le.fit_transform(X['enginetype'])
X['fuelsystem'] = le.fit_transform(X['fuelsystem'])

X.head()

cylinder_map = {'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12}
X['cylindernumber'] = X['cylindernumber'].map(cylinder_map)

X.head()

"""## Spilting dataset into training and test"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

"""## Fit the model"""

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""## Accuracy of model"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('R squared is: {} %'.format(np.floor(r_squared*100)))
print('MSE is: {}'.format(mse))
print('MAE is: {}'.format(mae))

"""## Visualization"""

plt.scatter(y_test, y_pred)
plt.show()
