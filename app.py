# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:05:52 2021

@author: Nicolò
"""

# Importing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn import  linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from termcolor import colored

import plotly.express as px

# Files reading
p2017=pd.read_csv('assets/IST_North_Tower_2017_Ene_cons.csv')
p2018=pd.read_csv('assets/IST_North_Tower_2018_Ene_Cons.csv')
weather0=pd.read_csv('assets/IST_meteo_data_2017_2018_2019.csv')
holiday0=pd.read_csv('assets/holiday_17_18_19.csv')

# Info
p2017.info()
p2018.info()
weather0.info()
holiday0.info()

# Merging 2017 and 2018
p1718=pd.concat([p2017,p2018])
print(p1718)

# Same name for Date and time columns
p1718=p1718.rename(columns={'Date_start':'Date and time'})
weather=weather0.rename(columns={'yyyy-mm-dd hh:mm:ss':'Date and time'})
holiday=holiday0.rename(columns={'Date':'Date and time'})

# Datetime format
p1718['Date and time']=pd.to_datetime(p1718['Date and time'],format=('%d-%m-%Y %H:%M'))
weather['Date and time']=pd.to_datetime(weather['Date and time'],format=('%Y-%m-%d %H:%M:%S'))
holiday['Date and time']=pd.to_datetime(holiday['Date and time'],format=('%d.%m.%Y'))

# Datetime as a index
p1718=p1718.set_index(pd.DatetimeIndex(p1718['Date and time']))
weather=weather.set_index(pd.DatetimeIndex(weather['Date and time']))
holiday=holiday.set_index(pd.DatetimeIndex(holiday['Date and time']))

# Drop the columns in excess
p1718=p1718.drop(columns='Date and time') # No NaN
weather=weather.drop(columns='Date and time') # Missing data
holiday=holiday.drop(columns='Date and time') # No NaN

# Only value of integer hours
weather=weather.resample('H').mean()

# Showing NaN
print(p1718[p1718.isnull().any(axis = 'columns')]) # No NaN
print(weather[weather.isnull().any(axis = 'columns')]) # Missing data!
print(holiday[holiday.isnull().any(axis = 'columns')]) # No NaN

# Final merging
data = pd.merge(p1718,weather,on='Date and time')
data = data.dropna() # Dropping weather NaN (before merging with holiday in order to avoid dropping non-holidays)
data=pd.concat([data,holiday])
data[['Holiday']] = data[['Holiday']].fillna(value=0) # I put 0 whenever it is not a holiday
# print(data['Holiday'])

# Remove 2019 data
dataclean = data.iloc[:15384,:] 
# print(dataclean.tail())

# Renaming columns
dataclean.rename(columns = {'Power_kW': 'Power [kW]', 'temp_C': 'Temperature [°C]', 'HR': 'HR [%]', 'windSpeed_m/s':'Wind Speed [m/s]', 'windGust_m/s': 'Wind Gust [m/s]', 'solarRad_W/m2': 'Solar Radiation [W/m2]', 'pres_mbar': 'Pressure [mbar]', 'rain_mm/h': 'Rain [mm/h]', 'rain_day': 'Rainday'}, inplace = True)
print(dataclean.info())

# Sorting power in descending and ascending order
datasortP1 = dataclean.sort_values(by = 'Power [kW]', ascending = False)
datasortP2 = dataclean.sort_values(by = 'Power [kW]', ascending = True)
# print(datasortP1 [:15])
# print(datasortP2 [:15])
# High values of power seem reasonable, whereas P = 0 probably correspond to a Wi-fi error in those hours

# Sorting Temperature
datasortT1 = dataclean.sort_values(by = 'Temperature [°C]', ascending = False)
datasortT2 = dataclean.sort_values(by = 'Temperature [°C]', ascending = True)
# print(datasortT1 [:15])
# print(datasortT2 [:15])
# Here I realized of all the missing data of the weather that in a first moment I didn't find

# Sorting Solar Radiation
datasortS1 = dataclean.sort_values(by = 'Solar Radiation [W/m2]', ascending = False)
datasortS2 = dataclean.sort_values(by = 'Solar Radiation [W/m2]', ascending = True)
# print(datasortS1['Solar Radiation [W/m2]'] [:15])
# print(datasortS2['Solar Radiation [W/m2]'] [:15])

# Graphs to visually check eventual problems
'''
plt.plot(dataclean['Power [kW]'])
plt.title('Power [kW]')
plt.show()
plt.plot(dataclean['Temperature [°C]'])         # Significant Missing data in the end of 2018
plt.title('Temperature [°C]')
plt.show()
plt.plot(dataclean['HR [%]'])                   # Significant Missing data in the end of 2018
plt.title('HR [%]')
plt.show()
plt.plot(dataclean['Wind Speed [m/s]'])         # Significant Missing data in the end of 2018
plt.title('Wind Speed [m/s]')
plt.show()
plt.plot(dataclean['Wind Gust [m/s]'])          # Significant Missing data in the end of 2018
plt.title('Wind Gust [m/s]')
plt.show()
plt.plot(dataclean['Pressure [mbar]'])          # Significant Missing data in the end of 2018
plt.title('Pressure [mbar]')
plt.show()
plt.plot(dataclean['Solar Radiation [W/m2]'])   # Significant Missing data in the end of 2018
plt.title('Solar Radiation [W/m2]')
plt.show()
plt.plot(dataclean['Rain [mm/h]'])              # Significant Missing data in the end of 2018
plt.title('Rain [mm/h]')
plt.show()
plt.plot(dataclean['Rainday'])                  # Significant Missing data in the end of 2018
plt.title('Rainday')
plt.show()
'''

# Removing lowest values
q1 = dataclean['Power [kW]'].quantile(0.25)
q3 = dataclean['Power [kW]'].quantile(0.75)
dataclean = dataclean[dataclean['Power [kW]'] > q1]
# Graph
f1=plt.plot(dataclean['Power [kW]'])
plt.title('Power [kW] (cleaned)')
plt.show()

# Introducing columns with Hours and Week Day
dataclean['Hour'] = dataclean.index.hour
dataclean['Week Day'] = dataclean.index.weekday

# Power of the prevoius hour
dataclean['Power-1'] = dataclean['Power [kW]'].shift(1)
dataclean = dataclean.dropna() # Drop the first raw because there we cannot have the power of the previous hour
# print(dataclean.info())
'''
# CLUSTERING

model = KMeans(n_clusters=2).fit(dataclean)
pred = model.labels_
dataclean['Clusters']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dataclean).score(dataclean) for i in range(len(kmeans))]
score
plt.plot(Nc,score) # Good scores from n° = 2 and so on
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# With n° clusters = 2 no significant results
# I only found 3 well defined clusters regarding Solar Radiation
# For the power, one cluster has low value, the other two are mixed
# 3 clusters of different value of solar radiation (As expected the highest focused in the middle of the day)
# I will analyze more features through the features selection methods

Colors= np.where(dataclean['Clusters']==0,'blue',np.where(dataclean['Clusters']==1,'red','green'))
T_S=dataclean.plot.scatter(x='Temperature [°C]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Power [kW]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Week Day',y='Hour',color=Colors)
P_S=dataclean.plot.scatter(x='Solar Radiation [W/m2]',y='Hour',color=Colors)

model = KMeans(n_clusters=3).fit(dataclean)
pred = model.labels_
dataclean['Clusters']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dataclean).score(dataclean) for i in range(len(kmeans))]
score

Colors= np.where(dataclean['Clusters']==0,'blue',np.where(dataclean['Clusters']==1,'red',np.where(dataclean['Clusters']==2,'green','yellow')))
T_S=dataclean.plot.scatter(x='Temperature [°C]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Power [kW]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Week Day',y='Hour',color=Colors)
P_S=dataclean.plot.scatter(x='Solar Radiation [W/m2]',y='Hour',color=Colors)

model = KMeans(n_clusters=4).fit(dataclean)
pred = model.labels_
dataclean['Clusters']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dataclean).score(dataclean) for i in range(len(kmeans))]
score

Colors= np.where(dataclean['Clusters']==0,'blue',np.where(dataclean['Clusters']==1,'red',np.where(dataclean['Clusters']==2,'green',np.where(dataclean['Clusters']==3,'yellow','pink'))))
T_S=dataclean.plot.scatter(x='Temperature [°C]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Power [kW]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Week Day',y='Hour',color=Colors)
P_S=dataclean.plot.scatter(x='Solar Radiation [W/m2]',y='Hour',color=Colors)

model = KMeans(n_clusters=5).fit(dataclean)
pred = model.labels_
dataclean['Clusters']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dataclean).score(dataclean) for i in range(len(kmeans))]
score

Colors= np.where(dataclean['Clusters']==0,'blue',np.where(dataclean['Clusters']==1,'red',np.where(dataclean['Clusters']==2,'green',np.where(dataclean['Clusters']==3,'yellow',np.where(dataclean['Clusters']==4,'pink','black')))))
T_S=dataclean.plot.scatter(x='Temperature [°C]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Power [kW]',y='Solar Radiation [W/m2]',color=Colors)
P_S=dataclean.plot.scatter(x='Week Day',y='Hour',color=Colors)
P_S=dataclean.plot.scatter(x='Solar Radiation [W/m2]',y='Hour',color=Colors)
P_S=dataclean.plot.scatter(x='Solar Radiation [W/m2]',y='Hour',color=Colors)

# FEATURES SELECTION

# Inputs and Outputs
M = dataclean.values
Y = M[:,0]
X = M[:,[1,2,3,4,5,6,7,8,9,10,11,12]] 

# FILTER METHOD 1
features = SelectKBest(k=2,score_func=f_regression) 
fit = features.fit(X,Y)
print('Filter method 1 scores: %s' % (fit.scores_)) # Power-1, Solar Radiation, Weekday, Temperature
# WRAPPING METHOD 1
model = LinearRegression() 
rfe=RFE(model,4)
rfe2=RFE(model,5) 
fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)
print( "Wrapping Method 4 Features: %s" % (fit.ranking_)) # Wind Speed, Wind Gust, Rain, Hour
print( "Wrapping Method 5 Features: %s" % (fit2.ranking_)) # Wind Speed, Wind Gust, Rain, Hour, Power-1

# ENSEMBLE METHOD 1
model = RandomForestRegressor()
model.fit(X, Y)
print('Ensemble method 1: %s' % (model.feature_importances_)) # Power-1, Hour, Solar Radiation, Week Day, Temperature

# Trying with different features
dataclean['Heating Degree Days'] = np.maximum(0,dataclean['Temperature [°C]']-16)
dataclean['Days times Holiday'] = dataclean['Week Day']*(1-dataclean['Holiday'])
# print(dataclean.info())

M = dataclean.values
Y = M[:,0]
X = M[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]] 

# FILTER METHOD 2
features = SelectKBest(k=2,score_func=f_regression) 
fit = features.fit(X,Y)
print('Filter method 2 scores: %s' % (fit.scores_)) # Power-1, Solar Radiation, WeekDay, Days times Holiday, Heating Degree Days

# WRAPPING METHOD 2
model=LinearRegression() # LinearRegression Model as Estimator
rfe=RFE(model,4)# using 2 features
rfe2=RFE(model,5) # using 3 features
fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)
print( "Wrapping Method 4 Features: %s" % (fit.ranking_)) # Wind Speed, Wind Gust, Rain, Hour
print( "Wrapping Method 5 Features: %s" % (fit2.ranking_)) # Wind Speed, Wind Gust, Rain, Hour, Power-1

# ENSEMBLE METHOD 2
model = RandomForestRegressor()
model.fit(X, Y)
print('Ensemble method 2: %s' % (model.feature_importances_)) # Power-1, Hour, Solar Radiation, Temperature, Days times Holiday

# Wrapping method doesn't appear reliable with these data. In fact, wind speed and wind gust are found relevant only by using
# this method and Power-1 is not highlighted as important.
# It is paramount to take into account a feature regarding temperature becausae in the North Tower
# the heating system is based on natural gas. Thus, the electric power is consumed for cooling (with high T) and
# not for heating (with low T). Basing on the methods utilized, Temperature and Heating Degree Days are chosen.
# Therefore the selected features are:
#    Power-1
#    Temperature
#    Solar radiation
#    Hour
#    Heating Degree Days
#    Days times Holiday

# Columns dropping
datamodel =  dataclean.drop(columns=['HR [%]','Wind Speed [m/s]','Wind Gust [m/s]','Pressure [mbar]','Rain [mm/h]','Rainday','Holiday','Week Day'])
print(datamodel.info())

# REGRESSION MODELS
M = datamodel.values 
Y = M[:,0]
X = M[:,[1,2,3,4,5,6]]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X,Y)

# LINEAR REGRESSION
LR = linear_model.LinearRegression()
LR.fit(X_train,y_train)
y_pred_LR = LR.predict(X_test)
# Graphs
plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_LR[1:200], marker='*', markersize=6)
plt.title('Linear Regression')
plt.show()
plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_LR[1:200], marker='*', markersize=6)
plt.title('Linear Regression')
plt.show()
ff = plt.scatter(y_test,y_pred_LR)
plt.title('Linear Regression')
plt.show()
# Errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(colored('LINEAR REGRESSION METHOD ERRORS:','red'))
print('LR Mean absolut error: %s' % (MAE_LR))
print('LR Mean squared error: %s' % (MSE_LR))
print('LR Root mean squared error: %s' % (RMSE_LR))
print('LR Coefficient of variation of Root mean squared error: %s' % (cvRMSE_LR))

# SUPPORT VECTOR REGRESSOR
Sc_X = StandardScaler()
Sc_y = StandardScaler()
X_train_SVR = Sc_X.fit_transform(X_train)
y_train_SVR = Sc_y.fit_transform(y_train.reshape(-1,1))
SVR = SVR(kernel='rbf') # With poly and sigmoid bigger errors
SVR.fit(X_train_SVR,y_train_SVR)
y_pred_SVR = SVR.predict(Sc_X.fit_transform(X_test))
y_test_SVR=Sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=Sc_y.inverse_transform(y_pred_SVR) # Coming back to the original values
# Graphs
plt.plot(y_test_SVR[1:200],marker='o', markersize=6)
plt.plot(y_pred_SVR[1:200],marker='*', markersize=6)
plt.title('Support Vector Regressor (Scaled)')
plt.show()
plt.plot(y_test[1:200],marker='o', markersize=6)
plt.plot(y_pred_SVR2[1:200],marker='*', markersize=6)
plt.title('Support Vector Regressor')
plt.show()
ff2 = plt.scatter(y_test,y_pred_SVR)
plt.title('Support Vector Regressor.')
plt.show()
# Errors
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(colored('SUPPORT VECTOR REGRESSOR METHOD ERRORS:','red'))
print('SVR Mean absolut error: %s' % (MAE_SVR))
print('SVR Mean squared error: %s' % (MSE_SVR))
print('SVR Root mean squared error: %s' % (RMSE_SVR))
print('SVR Coefficient of variation of Root mean squared error: %s' % (cvRMSE_SVR))

# DECISION TREE
DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred_DT = DT.predict(X_test)

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_DT[1:200],marker='*', markersize=6)
plt.title('Decision Tree')
plt.show()

plt.scatter(y_test,y_pred_DT)
plt.title('Decision Tree')
plt.show()

# Errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(colored('DECISION TREE METHOD ERRORS:','red'))
print('DT Mean absolut error: %s' % (MAE_DT))
print('DT Mean squared error: %s' % (MSE_DT))
print('DT Root mean squared error: %s' % (RMSE_DT))
print('DT Coefficient of variation of Root mean squared error: %s' % (cvRMSE_DT))

# RANDOM FOREST
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF = RandomForestRegressor(**parameters)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_RF[1:200], marker='*', markersize=6)
plt.title('Random Forest')
plt.show()

plt.scatter(y_test,y_pred_RF)
plt.title('Random Forest')
plt.show()

# Errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(colored('RANDOM FOREST METHOD ERRORS:','red'))
print('RF Mean absolut error: %s' % (MAE_RF))
print('RF Mean squared error: %s' % (MSE_RF))
print('RF Root mean squared error: %s' % (RMSE_RF))
print('RF Coefficient of variation of Root mean squared error: %s' % (cvRMSE_RF))

# UNIFORMIZING
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST UNIFORMIZED
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RFU = RandomForestRegressor(**parameters)
RFU.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RFU = RFU.predict(X_test_scaled)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_RFU[1:200], marker='*', markersize=6)
plt.title('Random Forest Uniformized')
plt.show()

plt.scatter(y_test,y_pred_RFU)
plt.title('Random Forest Uniformized')
plt.show()

# Errors
MAE_RFU=metrics.mean_absolute_error(y_test,y_pred_RFU) 
MSE_RFU=metrics.mean_squared_error(y_test,y_pred_RFU)  
RMSE_RFU= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RFU))
cvRMSE_RFU=RMSE_RFU/np.mean(y_test)
print(colored('RANDOM FOREST UNIFORMIZED METHOD ERRORS:','red'))
print('RFU Mean absolut error: %s' % (MAE_RFU))
print('RFU Mean squared error: %s' % (MSE_RFU))
print('RFU Root mean squared error: %s' % (RMSE_RFU))
print('RFU Coefficient of variation of Root mean squared error: %s' % (cvRMSE_RFU))

#GRADIENT BOOSTING
GB = GradientBoostingRegressor()
GB.fit(X_train, y_train)
y_pred_GB =GB.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_GB[1:200], marker='*', markersize=6)
plt.title('Gradient Boosting')
plt.show()

plt.scatter(y_test,y_pred_GB)
plt.title('Gradient Boosting')
plt.show()

# Errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(colored('GRADIENT BOOSTING METHOD ERRORS:','red'))
print('GB Mean absolut error: %s' % (MAE_GB))
print('GB Mean squared error: %s' % (MSE_GB))
print('GB Root mean squared error: %s' % (RMSE_GB))
print('GB Coefficient of variation of Root mean squared error: %s' % (cvRMSE_GB))

# EXTREME GRADIENT BOOSTING
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)
XGB = XGBRegressor()
XGB.fit(X_train, y_train)
y_pred_XGB =XGB.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_XGB[1:200], marker='*', markersize=6)
plt.title('Extreme Gradient Boosting')
plt.show()

plt.scatter(y_test,y_pred_XGB)
plt.title('Extreme Gradient Boosting')
plt.show()

# Errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(colored('EXTREME GRADIENT BOOSTING METHOD ERRORS:','red'))
print('XGB Mean absolut error: %s' % (MAE_XGB))
print('XGB Mean squared error: %s' % (MSE_XGB))
print('XGB Root mean squared error: %s' % (RMSE_XGB))
print('XGB Coefficient of variation of Root mean squared error: %s' % (cvRMSE_XGB))

# BOOTSTRAPPING
BT = BaggingRegressor()
BT.fit(X_train, y_train)
y_pred_BT =BT.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_XGB[1:200], marker='*', markersize=6)
plt.title('Bootstrapping')
plt.show()

ff7 = plt.scatter(y_test,y_pred_BT)
plt.title('Bootstrapping')
plt.show()
# Errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(colored('BOOTSTRAPPING METHOD ERRORS:','red'))
print('BT Mean absolut error: %s' % (MAE_BT))
print('BT Mean squared error: %s' % (MSE_BT))
print('BT Root mean squared error: %s' % (RMSE_BT))
print('BT Coefficient of variation of Root mean squared error: %s' % (cvRMSE_BT))

# Now I try to change parameters in some of the models with the lowest errors

# RANDOM FOREST 2
parameters = {'bootstrap': True,
              'min_samples_leaf': 4,
              'n_estimators': 300, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 50,
              'max_leaf_nodes': None}
RF = RandomForestRegressor(**parameters)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_RF[1:200], marker='*', markersize=6)
plt.title('Random Forest')
plt.show()

plt.scatter(y_test,y_pred_RF)
plt.title('Random Forest')
plt.show()

# Errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(colored('RANDOM FOREST 2 METHOD ERRORS:','red'))
print('RF Mean absolut error: %s' % (MAE_RF))
print('RF Mean squared error: %s' % (MSE_RF))
print('RF Root mean squared error: %s' % (RMSE_RF))
print('RF Coefficient of variation of Root mean squared error: %s' % (cvRMSE_RF))


# EXTREME GRADIENT BOOSTING
#parameters = {'n_estimators': 800, 'max_depth': 10, 'min_samples_split': 3,
#          'learning_rate': 0.02, 'loss': 'ls'}
parameters = {'n_estimators': 600, 'max_depth': 10,
          'learning_rate': 0.02}
XGB = XGBRegressor(**parameters)
XGB.fit(X_train, y_train)
y_pred_XGB =XGB.predict(X_test)
# Graphs

plt.plot(y_test[1:200], marker='o', markersize=6)
plt.plot(y_pred_XGB[1:200], marker='*', markersize=6)
plt.title('Extreme Gradient Boosting')
plt.show()

ff = plt.scatter(y_test,y_pred_XGB)
plt.title('Extreme Gradient Boosting')
plt.show()

# Errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(colored('EXTREME GRADIENT BOOSTING 2 METHOD ERRORS:','red'))
print('XGB Mean absolut error: %s' % (MAE_XGB))
print('XGB Mean squared error: %s' % (MSE_XGB))
print('XGB Root mean squared error: %s' % (RMSE_XGB))
print('XGB Coefficient of variation of Root mean squared error: %s' % (cvRMSE_XGB))
'''
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config ['suppress_callback_exceptions'] = True

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo2.png')),
    html.H2('Energy Services Project 2- IST - North Tower'),
    html.B('By Nicolò Italiano - ist198008'),
    dcc.Tabs(id='tabs', value='tab-0', children=[
        dcc.Tab(label='Raw data', value='tab-0'),
        dcc.Tab(label='Data cleaning', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature selection', value='tab-3'),
        dcc.Tab(label='Forecasts', value='tab-4'),
        ]),
    html.Div(id='tabs-content'),
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))


def render_content(tab):
    if tab == 'tab-0':
        return html.Div([
            html.H3('Raw data'),
            html.B('The following data from North Tower of IST have been provided:'),
            dcc.Dropdown(
                id='Dropdown',
                options=[
                    {'label':'Power Consumption 2017','value':1},
                    {'label':'Power Consumption 2018','value':2},
                    {'label':'Weather','value':3},
                    {'label':'Holiday','value':4},
                ],
                value=1
                ),
            html.Div(id='tables'),
            ])
    elif tab == 'tab-1':
        return html.Div([
            html.H3('Data cleaning'),
            html.B('The Power data of 2017 and 2018 have been merged with Weather and Holiday data. Afterwards the NaN and the Power data below the 25% quantile have been removed. Each feature is depicted below throughout the period taken into account:'),
            html.Label('Select the feature:'), 
            dcc.RadioItems(
                id='radio',
                options=[
                    {'label':'Power [kW]','value':1},
                    {'label':'Temperature [°C]','value':2},
                    {'label':'HR [%]','value':3},
                    {'label':'Wind Speed [m/s]','value':4},
                    {'label':'Wind Gust [m/s]','value':5},
                    {'label':'Pressure [mbar]','value':6},
                    {'label':'Solar Radiation [W/m2]','value':7},
                    {'label':'Rain [mm/h]','value':8},
                    {'label':'Rainday','value':9},
                    {'label':'Holiday','value':10},
                ],
                value=1
                ),
            html.Div(id='Graph'),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Clustering'), 
            html.Label('Number of k means'),
            dcc.Slider(
                min=2,
                max=5,
                id='k',
                value=2,
                marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(2, 6)}
            ),
            html.Div(id='Slider'),
            html.B('*Only the graph that clearly differentiate the clusters have been depicted'),
        ]),
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Feature selection method'),
            html.B('In order to select the features to be used in the forecasts, there have been applied 3 methods. For each method, the best 4 features detected are shown. After a first analysis, the methods are reapeted with 2 new features: HDD and Days times Holiday. HDD corresponds to the Heating Degree Days while Days times Holiday has been introduced in order to compress these 2 features into one. Below it is possible to select each method and see the 4 more relevant features:'),
            dcc.Dropdown(
                id='Features',
                options=[
                    {'label':'Filter Method','value':1},
                    {'label':'Wrapping Method','value':2},
                    {'label':'Ensemble Method','value':3},
                    {'label':'Filter Method (with "HDD" and "Days x Holiday")','value':4},
                    {'label':'Wrapping Method (with "HDD" and "Days x Holiday")','value':5},
                    {'label':'Ensemble Method (with "HDD" and "Days x Holiday")','value':6},
                ],
                value=1
                ),
            html.H5('Best 4 features:'),
            html.Div(id='List'),
            html.H5('Conclusions'),
            html.B("Wrapping method doesn't appear reliable with these data. In fact, wind speed and wind gust are found relevant only by using this method and Power-1 is not highlighted as important. It is paramount to take into account a feature regarding temperature because in the North Tower the heating system is based on natural gas. This implies that the electric power is consumed for cooling (with high T) and not for heating (with low T). Therefore, basing on the methods utilized, Temperature and Heating Degree Days are chosen. The selected features are:"),
            dcc.Markdown('''Power-1, 
            Temperature,
            Solar radiation,
            Hour,
            HDD,
            Days times Holiday
                         '''),
#    Power-1
#    Temperature
#    Solar radiation
#    Hour
#    Heating Degree Days
#    Days times Holiday')
        ]),        
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Regression Models'),
            html.Label('Select the Regression Model:'),
            dcc.Dropdown(
                id='Regression',
                options=[
                    {'label':'Linear Regression','value':1},
                    {'label':'Support vector regression','value':2},
                    {'label':'Decision Tree','value':3},
                    {'label':'Random Forest','value':4},
                    {'label':'Random Forest Uniformized','value':5},
                    {'label':'Gradient Boosting','value':6},
                    {'label':'Extreme Gradient Boosting','value':7},
                    {'label':'Bootstrapping','value':8},
                ],
                value=1
                ),
            html.Div(id='Modelgraph'),
        ]),

@app.callback(Output('tables', 'children'), 
              Input('Dropdown', 'value'))

def render_table(Dropdown):
    if Dropdown == 1:
        return generate_table(p2017,max_rows=15)
    elif Dropdown == 2:
        return generate_table(p2018,max_rows=15)
    elif Dropdown == 3:
        return generate_table(weather0,max_rows=15)
    elif Dropdown == 4:
        return generate_table(holiday0,max_rows=15)

@app.callback(Output('Graph', 'children'), 
              Input('radio', 'value'))

def render_graph(radio):
    if radio == 1:
        return dcc.Graph(figure=px.line(dataclean, y='Power [kW]')),
    elif radio == 2:
        return dcc.Graph(figure=px.line(dataclean, y='Temperature [°C]')),
    elif radio == 3:
        return dcc.Graph(figure=px.line(dataclean, y='HR [%]')),
    elif radio == 4:
        return dcc.Graph(figure=px.line(dataclean, y='Wind Speed [m/s]')),
    elif radio == 5:
        return dcc.Graph(figure=px.line(dataclean, y='Wind Gust [m/s]')),
    elif radio == 6:
        return dcc.Graph(figure=px.line(dataclean, y='Pressure [mbar]')),
    elif radio == 7:
        return dcc.Graph(figure=px.line(dataclean, y='Solar Radiation [W/m2]')),
    elif radio == 8:
        return dcc.Graph(figure=px.line(dataclean, y='Rain [mm/h]')),
    elif radio == 9:
        return dcc.Graph(figure=px.line(dataclean, y='Rainday')),
    elif radio == 10:
        return dcc.Graph(figure=px.line(dataclean, y='Holiday')),

@app.callback(Output('Slider', 'children'), 
              Input('k', 'value'))

def render_graphs(k):
    
    if k == 2:
        return html.Div([html.Img(src='assets/2.1.png'),html.Img(src='assets/2.2.png'),html.Img(src='assets/2.3.png'),html.Img(src='assets/2.4.png')],style={'ColumnsCount':2})
    elif k == 3:
        return html.Div([html.Img(src='assets/3.1.png'),html.Img(src='assets/3.2.png'),html.Img(src='assets/3.3.png'),html.Img(src='assets/3.4.png')],style={'ColumnsCount':2})
    elif k == 4:
        return html.Div([html.Img(src='assets/4.1.png'),html.Img(src='assets/4.2.png'),html.Img(src='assets/4.3.png'),html.Img(src='assets/4.4.png')],style={'ColumnsCount':2})
    elif k == 5:
        return html.Div([html.Img(src='assets/5.1.png'),html.Img(src='assets/5.2.png'),html.Img(src='assets/5.3.png'),html.Img(src='assets/5.4.png')],style={'ColumnsCount':2})

    
@app.callback(Output('List', 'children'), 
              Input('Features', 'value'))

def render_list(Features):
    
    if Features == 1:
        return html.B('Power-1, Solar Radiation, Weekday, Temperature')
    elif Features == 2:
        return html.B('Wind Speed, Wind Gust, Rain, Hour')
    elif Features == 3:
        return html.B('Power-1, Hour, Solar Radiation, Week Day')
    elif Features == 4:
        return html.B('Power-1, Solar Radiation, WeekDay, Days x Holiday')
    elif Features == 5:
        return html.B('Wind Speed, Wind Gust, Rain, Hour')
    elif Features == 6:
        return html.B('Power-1, Hour, Solar Radiation, Temperature')

@app.callback(Output('Modelgraph', 'children'), 
              Input('Regression', 'value'))

def render_graphs2(Regression):
    
    if Regression == 1:
        return html.Div([html.Img(src='assets/LR_Scatter.png'),html.Img(src='assets/LR.png')],style={'ColumnsCount':2}),html.B('LR Mean absolut error: 18.36; LR Mean squared error: 784.59; LR Root mean squared error: 28.01; LR Coefficient of variation of Root mean squared error: 0.2101')
    elif Regression == 2:
        return html.Div([html.Img(src='assets/SVR_Scatter.png'),html.Img(src='assets/SVR.png')],style={'ColumnsCount':2}),html.B('SVR Mean absolut error: 11.35, SVR Mean squared error: 413.57, SVR Root mean squared error: 20.34, SVR Coefficient of variation of Root mean squared error: 0.1526')
    elif Regression == 3:
        return html.Div([html.Img(src='assets/DT_Scatter.png'),html.Img(src='assets/DT.png')],style={'ColumnsCount':2}),html.B('DT Mean absolut error: 10.16, DT Mean squared error: 486.55, DT Root mean squared error: 22.06, DT Coefficient of variation of Root mean squared error: 0.1654')
    elif Regression == 4:
        return html.Div([html.Img(src='assets/RF_Scatter.png'),html.Img(src='assets/RF.png')],style={'ColumnsCount':2}),html.B('RF Mean absolut error: 9.06, RF Mean squared error: 246.92, RF Root mean squared error: 15.71, RF Coefficient of variation of Root mean squared error: 0.1179')
    elif Regression == 5:
        return html.Div([html.Img(src='assets/RFU_Scatter.png'),html.Img(src='assets/RFU.png')],style={'ColumnsCount':2}),html.B('RFU Mean absolut error: 10.17, RFU Mean squared error: 282.94, RFU Root mean squared error: 16.82, RFU Coefficient of variation of Root mean squared error: 0.1262')
    elif Regression == 6:
        return html.Div([html.Img(src='assets/GB_Scatter.png'),html.Img(src='assets/GB.png')],style={'ColumnsCount':2}),html.B('GB Mean absolut error: 10.28, GB Mean squared error: 301.81, GB Root mean squared error: 17.37, GB Coefficient of variation of Root mean squared error: 0.1303')
    elif Regression == 7:
        return html.Div([html.Img(src='assets/XGB_Scatter.png'),html.Img(src='assets/XGB.png')],style={'ColumnsCount':2}),html.B('XGB Mean absolut error: 8.00, XGB Mean squared error: 244.54, XGB Root mean squared error: 15.64, XGB Coefficient of variation of Root mean squared error: 0.1173')
    elif Regression == 8:
        return html.Div([html.Img(src='assets/BT_Scatter.png'),html.Img(src='assets/BT.png')],style={'ColumnsCount':2}),html.B('BT Mean absolut error: 8.34, BT Mean squared error: 274.76, BT Root mean squared error: 16.58, BT Coefficient of variation of Root mean squared error: 0.1244')


if __name__ == '__main__':
    app.run_server(debug=True)