# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#cost function: sum(yhat - y)^2 

#imoprt data set
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#get train set and test test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # we create a linear regression model by our training dataset

#predicting the Test set result
y_pred = regressor.predict(X_test)


# Visualising the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()

# for test set -> visualising
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience(test Set)")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()


