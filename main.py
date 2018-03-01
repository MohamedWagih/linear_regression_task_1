# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:56:29 2018

@author: Mohamed Wagih
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


# Importing raw data
raw_data = pd.read_csv('abalone.data', sep=",", header=None)
col_names = pd.read_csv('abalone.domain', sep=":", header=None)
raw_data.columns = [col for col in col_names[0]]

# Save data
raw_data.to_csv("abalone_data.csv")

# Importing data after refiened
data = pd.read_csv("abalone_data.csv", index_col=0)

# Encode labels
le = LabelEncoder()
le.fit(data["sex "])
data["sex "] = le.transform(data["sex "])


# Features and output
x = data.iloc[:,0:8]
y = data.iloc[:,8]


# normalization
for col in list(x.columns):
    x[col] /= x[col].max()
#    x[col] = (x[col]-x[col].mean())/x[col].max()


# Generate polynomial features.
poly_x = PolynomialFeatures(degree=2, include_bias=False)
x = poly_x.fit_transform(x)


# Training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42, shuffle=False)
linReg = LinearRegression()
linReg.fit(x_train, y_train)
result = linReg.predict(x_test)
RMSE = np.sqrt(mean_squared_error(y_test,result))
print(RMSE)



