# -*- coding: utf-8 -*-
"""Iris_Classification_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gViGzdq1C8Co3BgYTTNl25Ulnoxvwa3P

# Iris Classification Project

## import libraries
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""## Import dataset"""

from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data[:, :2]
y = dataset.target

df = pd.DataFrame(dataset.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
df['Target'] = y
df.head(20)

df.describe()

pd.DataFrame(X).head(20)

print(y)

print(dataset.feature_names)

print(dataset.target_names)

df.shape

"""## Data Visualization"""

df.plot(kind='box', sharex=False, sharey=False)
plt.show()

df.hist()
plt.show()

pd.plotting.scatter_matrix(df)
plt.show()

"""## Spilting dataset into traing set and test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

"""## Training the model"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = []
models = []
results = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %.2f (%f)' % (name, cv_results.mean() * 100, cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

"""## Predicting the model"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

predictions = model.predict(X_test)

print("Accuracy Score: {} %".format(accuracy_score(y_test, predictions) * 100))
print("Confusion Matrix: {} %".format(confusion_matrix(y_test, predictions) * 100))
# print("Classification Report: {} %".format(classification_report(y_test, predictions) * 100))

"""## Visualization"""

h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class labels for the mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.viridis)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend()

plt.show()

# Define the mesh grid for plotting
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class labels for the mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a dataframe of the input data and predicted labels
df = sns.load_dataset("iris").iloc[:, :2]
df['species'] = dataset.target_names[y]
df['predicted_species'] = dataset.target_names[model.predict(X)]

# Plot the data points and decision boundary using Seaborn and Matplotlib
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# Define the mesh grid for plotting
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class labels for the mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a dataframe of the input data and predicted labels
df = sns.load_dataset("iris").iloc[:, :2]
df['species'] = dataset.target_names[y]
df['predicted_species'] = dataset.target_names[model.predict(X)]

# Set Seaborn style and create figure and axes objects
sns.set_style("white")
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data points and decision boundary using Seaborn and Matplotlib
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=ax)
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
ax.set_xlabel('Sepal length', fontsize=14)
ax.set_ylabel('Sepal width', fontsize=14)
ax.set_title('Logistic regression decision boundary for iris dataset', fontsize=16)

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], loc='upper left', fontsize=12)

plt.show()