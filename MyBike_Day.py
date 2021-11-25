# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:21:35 2021

@author: chrysmok
"""
# import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
# import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model as skl_lm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statistics as st
from scipy import linalg
from math import sqrt
plt.style.use('seaborn-white')
plt.style.use("seaborn")
pd.set_option('max_columns', 1000)# display in spyder console up to 1000 columns
# bikeDay = pd.read_csv('day.csv', usecols=[1,2,3,4])
bikeDay = pd.read_csv('day.csv')
df = pd.read_csv('Data/Boston.csv', index_col=0)
bikeDay.head()

iris = sns.load_dataset('iris') 
iris.head()

from IPython.core.display import display_html 
# display(HTML(\"<style>.container { width:80% !important; }</style>\"))
bikeDay.shape
bike=bikeDay
bike.size
bike.isnull().sum()
bike.isnull().sum()
bike.info()
bike.dtypes
bike.describe()
bike.nunique()
# Learning Outcome:Except one column
bike_dup = bike.copy()
bike_dup.shape
#Create a copy of the  dataframe, without the 'instant' column
bike_dummy=bike.iloc[:,1:16]
for col in bike_dummy:
    print(bike_dummy[col].value_counts(ascending=False), )
    
bike.columns
# create categorical and then dummy variables
bike_new=bike[['season', 'yr', 'mnth', 'holiday', 'weekday',  'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']]
bike_new.info()
bike_new['season']=bike_new['season'].astype('category')
bike_new['weathersit']=bike_new['weathersit'].astype('category')
bike_new['mnth']=bike_new['mnth'].astype('category')
bike_new['weekday']=bike_new['weekday'].astype('category')
   

bike_new = pd.get_dummies(bike_new, drop_first= True)
bike_new.info()
bike_new.shape
##--------------------- split-holdout----------------
# We should specify 'random_state' so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(bike_new, train_size = 0.70, test_size = 0.30, random_state = 333)
df_train.columns
bike_num=df_train[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]
sns.pairplot(bike_num, diag_kind='kde')
plt.show()
# Box plot by catgories
#%% display ('ciao')
plt.figure(figsize=(25, 10))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike)
plt.subplot(2,3,3)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike)
plt.subplot(2,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike)
plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike)
plt.show()

####
for col in bike_dummy:
     print(bike_dummy[col].value_counts(ascending=False))
#bonjour
df_train.columns
#%% sns.pairplot(bike_num, diag_kind='kde')
#%% plt.show()
sns.pairplot(bike_num, diag_kind='auto')
plt.show()
#Correlation matrix
plt.figure(figsize = (25,20))
plt.figure(figsize = (35,30))
sns.heatmap(bike_new.corr(), annot = False, cmap="RdBu")
plt.show()
a=bike_new.corr();
sns.heatmap(a)
sns.heatmap(a,cmap="RdBu")
# =============================================================================
# scale
# =============================================================================

scaler = MinMaxScaler()
num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']
df_train[num_vars]= scaler.fit_transform(df_train[num_vars])
df_test[num_vars] = scaler.transform(df_test[num_vars])
# b.head()# ne se fait pas pour les matrices
df_train.loc[num_vars]= scaler.fit_transform(df_train[num_vars])
df_test.loc[num_vars] = scaler.transform(df_test[num_vars])
df_test.loc[num_vars] = scaler.fit_transform(df_test[num_vars])
df_train.head()
bike_dummy.info()
df_train.describe()
#%% Regression
y_train = df_train.pop('cnt')
X_train = df_train
df_test2=df_test.copy()
y_test=df_test.pop('cnt')
X_test=df_test
lm = LinearRegression()
lm.fit(X_train, y_train)
model=lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('TrueValues')
plt.ylabel('Predictions')
print ('Score:', model.score(X_test, y_test))
print ('Score:', model.score(X_train, y_train))
accuracy = metrics.r2_score(y_test, predictions)
#%% feature selection VIF ranking

rfe = RFE(lm, 15)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]#list of selected features
col
X_train.columns[~rfe.support_]
#check VIF
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = \"VIF\", ascending = False)
#apply rfe
X_train_rfe = X_train[col] #Creating X_test dataframe with RFE selected variables
print('vv')
X_test_rfe = X_test[col]
lm.fit(X_train_rfe, y_train)
model_rfe=lm.fit(X_train_rfe, y_train)
predictions_rfe = lm.predict(X_test_rfe)
plt.scatter(y_test, predictions_rfe)
plt.xlabel('TrueValues')
plt.ylabel('Predictions')
print ('Score:', model.score(X_test_rfe, y_test))
print ('Score:', model.score(X_train_rfe, y_train))
accuracy_rfe = metrics.r2_score(y_test, predictions_rfe)
#add a constant 
X_train_lm1 = sm.add_constant(X_train_rfe)
X_test_lm1 = sm.add_constant(X_test_rfe)
#%% OLS regression
lr1 = sm.OLS(y_train, X_train_lm1).fit()
predictions_OLS= lr1.predict(X_test_lm1)
plt.scatter(y_test, predictions_OLS)
plt.xlabel('TrueValues')
plt.ylabel('PredictionsOLS')
accuracy_lr1 = metrics.r2_score(y_test, predictions_OLS)
metrics.r2_score(y_train, lr1.predict(X_train_lm1))
lr1.rsquared
lr1.rsquared_adj
r2=metrics.r2_score(y_test, predictions_OLS)
adjusted_r_squared = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test_lm1.shape[1]-1)
# Residuals
res=y_train-lr1.predict(X_train_lm1)
sns.distplot((res))
# regression performance check the p-value of each variable and the global  F-stat
lr1.params
print(lr1.summary())
#remvove a variable
X_train_new = X_train_rfe.drop(['atemp'], axis = 1)
X_train_lm2 = sm.add_constant(X_train_new)
X_test_new = X_test_rfe.drop(['atemp'], axis = 1)
X_test_lm2 = sm.add_constant(X_test_new)
lr2 = sm.OLS(y_train, X_train_lm2).fit()
predictions_OLS2= lr2.predict(X_test_lm2)
plt.scatter(y_test, predictions_OLS2)
plt.xlabel('TrueValues')
plt.ylabel('PredictionsOLS2')
lr2.params
print(lr2.summary())


