# -*- coding: utf-8 -*-

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot') #for visualization

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999,inplace=True) #fill missing values

print(df.head())

forecast_out = int(math.ceil(0.1*len(df))) # taking 10% of the data
print("forecast_out:"+str(forecast_out))
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())

X = np.array(df.drop(['label'],1)) #features
X = preprocessing.scale(X)  #if new records are added, they need to be preprocessed together with the previous data
                           #hence adding to computation time
X_lately = X[-forecast_out:]
X = X[:-forecast_out]



#X = X[:-forecast_out +1]
df.dropna(inplace=True) #drop rows with missing attribute values
y = np.array(df['label'])

print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)

#Support Linear Regression
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)

#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
clf = LinearRegression(n_jobs=10) #10 threads
#clf = LinearRegression(n_jobs=-1) #as many threads as possible
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f) #saving model

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)

print(accuracy)

#predicting values using trained model
forecast_set = clf.predict(X_lately)
print(forecast_set,accuracy, forecast_out)
df['Forecast'] = np.nan
  
  
#visualization
#setting future dates, for x axis
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix +one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix) #next_date is the index of the dataframe
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
        #sets all the array items as not a number, and sets the last element of array as the forecast value

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()