# -*- coding: utf-8 -*-
"""Preprocessing_Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Hl9P_6cD3jNwU2mCtaZHvi4Xqm_g971X

# Pre-Processing
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')

df

df.groupby('Country').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

df.plot(kind='scatter', x='Age', y='Salary', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

df.head()

df.tail()

df.info()

df.describe().T

df.shape

df.isnull().sum()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X.shape

y.shape

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

X

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])

X

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train

X_test

