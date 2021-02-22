##Scrape live dataset from Yahoo Finance webpage 

#Modules for scraping imported
import pandas as pd
import requests
from bs4 import BeautifulSoup

#URL of Webpage to be scraped 
url = 'https://finance.yahoo.com/quote/FB/history?p=FB'
#df stores the first dataset scraped from the webpage
df = pd.read_html(url)[0]
#Delete the last row of irrelevant data
df.drop(df.tail(1).index , inplace=True)
df.rename(columns = {'Adj Close**':'Adj. Close'}, inplace = True)
print(df.head())

#Modules for machine learning imported
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Take only the row 'Adj Close**' which is the independent variable
df = df[['Adj. Close']]
print(df.head())

#Module will predict 'n' days into the future
n = 1
#Create the row which will be the dependent variable
df['Prediction'] = df[['Adj. Close']].shift(-n) #Adj Close** column is shifted up by 'n' units
print(df.tail())

#X is the independent data set
X = np.array(df.drop(['Prediction'],1))
#Remove last 'n' rows from X
X = X[:-n]
print(X)

#y is the dependent data set
y = np.array(df['Prediction'])
#Remove the NaN values (i.e the last 'n' values in y)
y = y[:-n]
print(y)

#Data split to 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creation of Support Vector Machine Regressor Model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#Training
svr_rbf.fit(x_train, y_train)

#Testing 
#score returns the coefficient of determination R^2 of the prediction (range : 0 to 1)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Creation of Linear Regression  Model
lr = LinearRegression()
#Training
lr.fit(x_train, y_train)

#Testing 
#score returns the coefficient of determination R^2 of the prediction (range : 0 to 1)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# X_pred is the last 'n' rows of independent variable 'Adj. Close'
X_pred = np.array(df.drop(['Prediction'],1))[-n:]
print(X_pred)

# Linear regression model predictions for next 'n' days
lr_y_pred = lr.predict(X_pred)
print(lr_y_pred)

# Support vector regression model predictions for next 'n' days
svm_y_pred = svr_rbf.predict(X_pred)
print(svm_y_pred)

#Graph of y_pred on test data v/s actual y for Support Vector Regression model
import matplotlib.pyplot as plt
test_y_pred = svr_rbf.predict(X)
y = y.astype('float64')
print(test_y_pred)
print(y)
ax = plt.plot(test_y_pred)
p2 = plt.plot(y , color = 'Red')
plt.show()

#Graph of y_pred on test data v/s actual data for Linear Regression model
import matplotlib.pyplot as plt
test_y_pred = lr.predict(X)
y = y.astype('float64')
print(test_y_pred)
print(y)
ax = plt.plot(test_y_pred)
p2 = plt.plot(y , color = 'Red')
plt.show()
