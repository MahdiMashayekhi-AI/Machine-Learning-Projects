# -*- coding: utf-8 -*-
"""Feature_Scaling_Iris_Dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e4bwQ5Wbwn_JrB1u3_tUlXt11GBIqfvS
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

df.head(10)

df['target'] = data.target

df.head(10)

data.target_names

"""## MinMax Scaler"""

df.drop('target', axis=1, inplace=True)

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

ax = df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='*', label='Before Scaling', color='r')
df1.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='+', label='After Scaling', ax=ax)
plt.axhline(0, color='r', alpha=0.2)
plt.axvline(0, color='r', alpha=0.2)
plt.show()

"""## Standard Scaler"""

scaler = StandardScaler()
df1 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

ax = df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='*', label='Before Scaling', color='r')
df1.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='+', label='After Scaling', ax=ax)
plt.axhline(0, color='r', alpha=0.2)
plt.axvline(0, color='r', alpha=0.2)
plt.show()

"""## Robust Scaler"""

scaler = RobustScaler()
df1 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

ax = df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='*', label='Before Scaling', color='r')
df1.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', marker='+', label='After Scaling', ax=ax)
plt.axhline(0, color='r', alpha=0.2)
plt.axvline(0, color='r', alpha=0.2)
plt.show()

