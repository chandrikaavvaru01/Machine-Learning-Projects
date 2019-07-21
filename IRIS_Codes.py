# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 07:16:01 2019

@author: chandu
"""

##Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Importing dataset

dataset = pd.read_csv('iris.data', header = None)
X = dataset.iloc[:,0:2].values
Y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)
onehotencoder = OneHotEncoder(categorical_features = [3])
Y = onehotencoder.fit_transform(Y).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)










# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)
X_opt= X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y , exog = X_opt).fit()
regressor_OLS.summary()
## When seen summary of the output the P_Value of the X2 variable is very larger than 0.5 so,according to Backward elimination we remve the variable and remodel the regressor
## Removing the 1,2& 4 columns as they have high p values
X_opt= X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y , exog = X_opt).fit()
regressor_OLS.summary()










## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state = 0, multi_class = 'multinomial')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



## Plotting Purpose
## If needed to plot then no need split the data into train and test sets

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])




# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()









