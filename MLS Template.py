# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:03:16 2019

@author: mrbaloglu
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("""put your data file's name here including the type extension""" ) 
X = dataset.iloc[:, :-1].values#assuming the last column of data contains the varible to be predicted
y = dataset.iloc[:, """put the index of last column here(0)"""].values 

# Encoding categorical data if exists else comment this section out
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
#for each i = 0,1,2,3,... (i) means the same as the sentence it first appears with
X[:,"""put indexes of columns that contain categorical data(1)"""] = labelencoder.fit_transform(X[:,"""(1)"""])
onehotencoder = OneHotEncoder(categorical_features = ["""(1)"""]) 
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap if exists else comment this section
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set, if you have your own test set comment this section
#out and use X and y instead of X_train and y_train respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Building the optimal model using the backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]] #here you should input [0,1,2,...,n] where n is number of columns-1 in X
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#exclude a variable according to the data of p-values and and adjusted r-squared values
X_opt = X[:,[0,1,3,4,5]] #this is shown as an example
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#continue to do this until you decide to stop
X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

#predict the results
regressor.fit(X_train2, y_train2)
y_pred2 = regressor.predict(X_test2)