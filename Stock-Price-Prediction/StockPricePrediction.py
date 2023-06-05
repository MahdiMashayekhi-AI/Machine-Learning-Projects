'''
Created by Mahdi Mashayekhi
Email : MahdiMashayekhi.ai@gmail.com
Github : https://github.com/MahdiMashayekhi-AI
Site : http://mahdimashayekhi.gigfa.com
YouTube : https://youtube.com/@MahdiMashayekhi
Twitter : https://twitter.com/Mashayekhi_AI
LinkedIn : https://www.linkedin.com/in/mahdimashayekhi/

'''

# Import Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data Preparation or EDA


def prepare_data(df, forecast_col, forecast_out, test_size):
    # creating new column called label with the last 5 rows are nan
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    # creating the column i want to use later in the predicting method
    X_lately = X[-forecast_out:]
    # X that will contain the training and testing
    X = X[:-forecast_out]
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)                           # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


df = pd.read_csv("prices.csv")  # Import your prices file from your computer
df = df[df.symbol == "GOOG"]

forecast_col = 'close'
forecast_out = 5
test_size = 0.2

# Applying for Stock Price Prediction
# Split data to tarin and test
# calling the method were the cross validation and data preperation is in
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(
    df, forecast_col, forecast_out, test_size)
learner = LinearRegression()  # initializing linear regression model
learner.fit(X_train, Y_train)  # training the linear regression model

# Prdict or test
score = learner.score(X_test, Y_test)  # testing the linear regression model
# set that will contain the forecasted data
forecast = learner.predict(X_lately)
response = {}  # creting json object
response['test_score'] = score
response['forecast_set'] = forecast

print(response)
