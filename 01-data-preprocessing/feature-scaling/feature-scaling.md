# Feature Scaling

Feature scaling is a technique to normalize the range of independent variables or features in your dataset.

## What is feature scaling?

Oh, right, first look at the table below to see the dataset.

| Age | Role_Level | Team_Size | Company_Value | Salary |
|-----|------------|-----------|----------------|--------|
| 24  | 1          | 4         | 3.5            | 38000     |
| 31  | 3          | 10        | 20.0           | 72000     |
| 28  | 2          | 6         | 8.0            | 55000     |
| 45  | 4          | 15        | 150.0          | 150000    |
| 35  | 3          | 8         | 25.0           | 80000     |
| 29  | 2          | 7         | 10.0           | 62000     |
| 39  | 4          | 12        | 100.0          | 120000    |
| 26  | 1          | 5         | 4.2            | 42000     |
| 50  | 5          | 20        | 200.0          | 180000    |
| 32  | 3          | 9         | 18.0           | 70000     |


In this example you can see that Salary column is so bigger than other columns and it's not good for models, so we should apply feature scaling technique on this dataset to bring all columns on a same unit.

## Why feature scaling is important?

Many machine learning algorithms perform better or coverage faster when the data is scaled. This is especially true for algorithms that use distance-base metrics, such as:

- K Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- K-Means Clustring
- Gradiant Descent-based models (like Logistic Regression)

If features have very diffrent ranges, the model may give more importance to features with larger values, which can lead to poor performance!

## Feature scaling methods:

- MinMax Scaler
- Standard Scaler
- Robust Scaler

### 1: MinMaxScaler

The Min-Max Scaler is a popular data normalization technique used in machine learning to transform features so that they fit within a specific range, usually [0, 1]. This helps machine learning models train more effectively and obtain more generalized values for coefficients and intercepts.

**Example:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```

#### Formula for MinMax
![image](https://miro.medium.com/v2/resize:fit:640/format:webp/1*ye1I00S61GqpR34ABZZFLQ.png)

### 2: Standard Scaler

StandardScaler standardizes features by removing the mean and scaling to unit variance. It transforms the data to have a mean of 0 and a standard deviation of 1. This process is also known as z-score normalization.

**Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

#### Formula for Standard
![image](https://journaldev.nyc3.cdn.digitaloceanspaces.com/2020/10/Standardization.png)

### 3: Robust Scaler

A robust scaler is a feature scaling technique that is less sensitive to outliers than methods like StandardScaler. It uses the median and interquartile range (IQR) instead of the mean and standard deviation, making it more robust to extreme values.

**Example:**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)
```

#### Formula for Robust
![image](https://miro.medium.com/v2/resize:fit:1400/1*I4KiunB5J6evG6IGcZ8j9w.png)

## When to Scale
1. Scale features before training the model.

2. You should not scale target variables in supervised learning.

3. Always apply the same scaler on training and test data (fit on training, transform both).