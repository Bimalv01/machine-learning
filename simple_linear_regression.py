# -*- coding: utf-8 -*-
"""Simple Linear Regression.ipynb

Automatically generated by Colaboratory.

"""# **Import Libraries**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""# **Import Dataset**"""

dataset=pd.read_csv('D:\mca sem 2\ml\SD (4).csv')

dataset

X = dataset.iloc[:, 0:1].values

X

y = dataset.iloc[:, 1:2].values

y

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.NaN,strategy='mean')
X1=imputer.fit_transform(X)
y1=imputer.fit_transform(y)

y1

"""# **Splitting the dataset into train and test set**"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3, random_state=0)

X_train

y_train

"""# **Fit the model to the training set**"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

y_pred

y_test

regressor.score(X1,y1)

predo=regressor.predict([[1.1]])

predo

plt.scatter(X_train,y_train, color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test, color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()