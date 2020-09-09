import pandas as pd
import numpy as np
#import geopandas as gpd
import matplotlib

import matplotlib.pyplot as plt
#import seaborn as sns
import statsmodels.api as sm
#sns.set()
#import plotly.express as px
#from plotly.offline import iplot
#import plotly.graph_objects as go
import re
import itertools
from dateutil import parser
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from datetime import datetime, date, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn import metrics


import time

import csv
import copy

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

#reference: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
#reference: https://www.kaggle.com/pbizil/random-forest-regression-for-time-series-predict
def add_population(row):
	row = str(row)
	#print(row)
	if row  == "Alabama":
		value = int(4903185)
		return value
	elif row == 'Alaska':
		value = int(731545)
		return value
	elif row == 'Arizona':
		value = int(7278717)
		return value
	elif row == 'Arkansas':
		value = int(3017804)
		return value
	elif row == 'California':
		value = int(39512223)
		return value
	elif row == 'Colorado':
		value = int(5758736)
		return value
	elif row == 'Connecticut':
		value = int(3565287)
		return value
	elif row == 'Delaware':
		value = int(973764)
		return value
	elif row == 'Florida':
		value = int(21477737)
		return value
	elif row == 'Georgia':
		value = int(10617423)
		return value
	elif row == 'Hawaii':
		value = int(1415872)
		return value
	elif row == 'Idaho':
		value = int(1787065)
		return value
	elif row == 'Illinois':
		value = int(12671821)
		return value
	elif row == 'Indiana':
		value = int(6732219)
		return value
	elif row == 'Iowa':
		value = int(3155070)
		return value
	elif row == 'Kansas':
		value = int(2913314)
		return value
	elif row == 'Kentucky':
		value = int(4467673)
		return value
	elif row == 'Louisiana':
		value = int(4648794)
		return value
	elif row == 'Maine':
		value = int(1344212)
		return value
	elif row == 'Maryland':
		value = int(6045680)
		return value
	elif row == 'Massachusetts':
		value = int(6892503)
		return value
	elif row == 'Michigan':
		value = int(9986857	)
		return value
	elif row == 'Minnesota':
		value = int(5639632)
		return value
	elif row == 'Mississippi':
		value = int(2976149	)
		return value
	elif row == 'Missouri':
		value = int(6137428	)
		return value
	elif row == 'Montana':
		value = int(1068778)
		return value
	elif row == 'Nebraska':
		value = int(1934408)
		return value
	elif row == 'Nevada':
		value = int(3080156)
		return value
	elif row == 'New Hampshire':
		value = int(1359711	)
		return value
	elif row == 'New Jersey':
		value = int(8882190)
		return value
	elif row== 'New Mexico':
		value = int(2096829)
		return value
	elif row == 'New York':
		value = int(19453561)
		return value
	elif row == 'North Carolina':
		value = int(10488084)
		return value
	elif row == 'North Dakota':
		value = int(762062)
		return value
	elif row == 'Ohio':
		value = int(11689100)
		return value
	elif row == 'Oklahoma':
		value = int(3956971)
		return value
	elif row == 'Oregon':
		value = int(4217737	)
		return value
	elif row == 'Pennsylvania':
		value = int(12801989)
		return value
	elif row == 'Rhode Island':
		value = int(1059361)
		return value
	elif row == 'South Carolina':
		value = int(5148714	)
		return value
	elif row == 'South Dakota':
		value = int(884659)
		return value
	elif row == 'Tennessee':
		value = int(6829174)
		return value
	elif row == 'Texas':
		value = int(28995881)
		return value
	elif row == 'Utah':
		value = int(3205958)
		return value
	elif row == 'Vermont':
		value = int(623989)
		return value
	elif row == 'Virginia':
		value = int(8535519)
		return value
	elif row == 'Washington':
		value = int(7614893)
		return value
	elif row == 'West Virginia':
		value = int(1792147)
		return value
	elif row == 'Wisconsin':
		value = int(5822434)
		return value
	elif row == 'Wyoming':
		value = int(578759)
		return value

def date_to_datetime(d):
    # You may need to modify this function, depending on your data types.
    #month = d.month
    #print(month)
    #print(len(d))

    if len(d) > 10:
    	m = re.search(r'(?<=-)\w+', d)
    	d = m.group(0)
    #m = re.sub("([0-9][0-9].)[0-9]+","" ,d)
    #print(m)
    #value = pd.to_datetime(d, format='%Y%m%d', errors='ignore')
    value = parser.parse(d)
    #print(value)
    #value = time.strptime(value, '%d.%m.%y')
    #print(value)
    value = '%04i-%02i-%02i' % (value.year, value.month, value.day)
    return value


def get_day(d):
	#if len(d) > 10:
	#	m = re.search(r'(?<=-)\w+', d)
	#	d = m.group(0)
	#m = re.sub("([0-9][0-9].)[0-9]+","" ,d)
	#print(m)
	#value = pd.to_datetime(d, format='%Y%m%d', errors='ignore')
	#value = parser.parse(d)
	#print(value)
	#value = time.strptime(value, '%d.%m.%y')
	#print(value)
	value = '%02i' % (d.day)
	return value

def get_month(d):
	#if len(d) > 10:
	#	m = re.search(r'(?<=-)\w+', d)
	#	d = m.group(0)
	#m = re.sub("([0-9][0-9].)[0-9]+","" ,d)
	#print(m)
	#value = pd.to_datetime(d, format='%Y%m%d', errors='ignore')
	#value = parser.parse(d)
	#print(value)
	#value = time.strptime(value, '%d.%m.%y')
	#print(value)
	value = '%02i' % (d.month)
	return value


def get_year(d):
	if len(d) > 10:
		m = re.search(r'(?<=-)\w+', d)
		d = m.group(0)
	#m = re.sub("([0-9][0-9].)[0-9]+","" ,d)
	#print(m)
	#value = pd.to_datetime(d, format='%Y%m%d', errors='ignore')
	value = parser.parse(d)
	#print(value)
	#value = time.strptime(value, '%d.%m.%y')
	#print(value)
	value = '%04i' % (value.year)
	return value

def sig_covid_bed(x):
	if (x > 10) and (x < 20):
		return '0'
	if x <= 10:
		return '1'
	if x >= 20:
		return '2'

def sig_covid_pop(x):
	if x >= 2.0:
		return '1'
	else:
		return '0'

def pop_amount(x):
	if x >= 5000000:
		return '2'
	elif (x < 5000000) and (x >= 1000000):
		return '1'
	elif x < 1000000:
		return '0'




df = pd.read_csv("covid19-NatEst.csv")

#for columns in df:
#	print(columns)

df2 = df[['State name', 'Day for which estimate is made','Hospital inpatient bed occupancy, percent estimate (percent of inpatient beds)', 'Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate', 'Number of patients in an inpatient care location who have suspected or confirmed COVID-19, percent estimate (percent of inpatient beds)']]
df2 = df2[df2['State name'] != 'United States']
df2 = df2[df2['State name'] != 'District of Columbia']
#df2 = df2[df2['State name'] == 'California']
df2 = df2.dropna()
df2['dateTime'] = df2['Day for which estimate is made'].apply(date_to_datetime)
df2['covid_bed_sig'] = df2['Number of patients in an inpatient care location who have suspected or confirmed COVID-19, percent estimate (percent of inpatient beds)'].apply(sig_covid_bed)
#print(df2.head())
print(len(df2))


df3 = pd.read_csv("us-states.csv")
print(len(df3))
df3 = df3[df3['state'] != 'District of Columbia']
#print(df3.head())

################# Regression ############################

########### prepare data with merging and groupby on data time #####################
new_df = df2.merge(df3, left_on=['dateTime','State name'], right_on = ['date','state'], how='left')
'''
new_df2 = new_df[['date','state', 'Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate','cases','deaths']]
new_df2 = new_df2[new_df2['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'] != 0]
new_df2.dropna()

new_df2['date'] = pd.to_datetime(new_df2['date'])
new_df2_test = new_df2.groupby('date')['cases'].sum().reset_index()
new_df2_grouped = new_df2.groupby('date')['cases'].mean().reset_index()
new_df2_grouped = new_df2_grouped.rename(columns={"Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate" : "bed amount"})
new_df2_grouped =  new_df2_grouped.set_index('date')
'''
new_df3 = new_df[['date','state', 'Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate','cases','deaths']]
new_df3  = new_df3 [new_df3['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'] != 0]
new_df3.dropna()
new_df3 = new_df3.rename(columns={"Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate" : "bed amount"})
new_df3 = new_df3[['date','state','bed amount', 'cases', 'deaths']]
new_df3['date'] = pd.to_datetime(new_df3['date'])
new_df3_grouped = new_df3.groupby(['date'])[['bed amount','cases']].sum().reset_index()
new_df3_grouped.index = new_df3_grouped.date
new_df3_grouped =new_df3_grouped.drop(columns = 'date')
#print(new_df3_grouped.tail())
new_df3 = new_df3_grouped 
#print(new_df3_grouped.head())

#create the lags for the bed amount to forecast
for i in range(1, 8):
    new_df3["lag_"+ str(i)] = new_df3['bed amount'].shift(i)

X1 = new_df3.copy()
X1 = X1.dropna()
X2 = X1.drop(['bed amount'], axis =1)
X = X2.iloc[:,0:]
y = X1['bed amount']

#scaled_x = StandardScaler()
#X = scaled_x.fit_transform(X)
#X = pd.DataFrame(X)
#print(X.head())

def TrainTestSplit(x, y):
	X_train, X_test = x.loc[x.index < '2020-07-01'] , x.loc[x.index >= '2020-07-01']
	Y_train, y_test = y.loc[y.index < '2020-07-01'] , y.loc[y.index >= '2020-07-01']
	return X_train, Y_train, X_test, y_test


X_train, y_train, X_test, y_test = TrainTestSplit(X,y)

regressor = LinearRegression(fit_intercept =False,normalize = True, n_jobs = 1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#print(X_test)

mae_scores = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("MAE SCORES: " + str(mae_scores))
print("R2 SCORE: " + str(score))


predictions_dates = []
start = pd.to_datetime('2020-07-01')
pred = []
day = 1

for i in range(7):
	date = start + timedelta(days = 1)
	predictions_dates.append(date)
	start = date

predictions_dates1 = []
start = pd.to_datetime('2020-06-20')
pred = []
day = 1

for i in range(12):
	date = start + timedelta(days = 1)
	predictions_dates1.append(date)
	start = date

all_dates = []
start = pd.to_datetime('2020-04-01')
for i in range(len(X) +1):
	date = start + timedelta(days = 1)
	all_dates.append(date)
	start = date

plt.figure(figsize=(20,8))
plt.title("Linear Regressor")
plt.plot_date(y= regressor.predict(X_test.values),x= predictions_dates, linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y = y_test.values,x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
#plt.plot_date(y = regressor.predict(x_valid.values),x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
plt.ylabel('amount')
plt.xlabel('date')
plt.legend(loc="best")
plt.show()


#traint test split the data to get a test set and training set and validation data ############
def TrainTestSplit(x, y):
	X_train, X_test = x.loc[x.index <= '2020-06-20'] , x.loc[(x.index <= '2020-07-01') & (x.index >= '2020-06-20')]
	Y_train, y_test = y.loc[y.index <= '2020-06-20'] , y.loc[(y.index <= '2020-07-01') & (y.index >= '2020-06-20')]
	x_valid, y_valid = x.loc[x.index >= '2020-07-01'], y.loc[y.index >= '2020-07-01'] ### validation dataset
	return X_train, Y_train, X_test, y_test, x_valid, y_valid
    
X_train, y_train, X_test, y_test, x_valid, y_valid = TrainTestSplit(X,y)


def TrainTestSplit(x, y):
	X_train, X_test = x.loc[x.index < '2020-07-01'] , x.loc[x.index >= '2020-07-01']
	Y_train, y_test = y.loc[y.index < '2020-07-01'] , y.loc[y.index >= '2020-07-01']
	return X_train, Y_train, X_test, y_test
    
X_train2, y_train2, X_test2, y_test2 = TrainTestSplit(X,y)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

########## regression models and fit #####################
regressor = LinearRegression(fit_intercept =False,normalize = True, n_jobs = 1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#print(X_test)

rfe = RandomForestRegressor(n_estimators= 600, max_depth = 8, min_samples_split = 20, random_state=0)
rfe = rfe.fit(X_train2, y_train2)
y_pred2 = rfe.predict(X_test2)

############ evaluating data #######################

mae_scores = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("MAE SCORES: " + str(mae_scores))
print("R2 SCORE: " + str(score))
mae_scores2 = mean_absolute_error(y_test2, y_pred2)

print("MAE SCORES: " + str(mae_scores2))
score2 = r2_score(y_test2, y_pred2)
print("R2 SCORE: " + str(score2))


predictions_dates = []
start = pd.to_datetime('2020-07-01')
pred = []
day = 1

for i in range(7):
	date = start + timedelta(days = 1)
	predictions_dates.append(date)
	start = date

predictions_dates1 = []
start = pd.to_datetime('2020-06-20')
pred = []
day = 1

for i in range(12):
	date = start + timedelta(days = 1)
	predictions_dates1.append(date)
	start = date

all_dates = []
start = pd.to_datetime('2020-04-01')
for i in range(len(X) +1):
	date = start + timedelta(days = 1)
	all_dates.append(date)
	start = date


############## plot data #####################
plt.figure(figsize=(20,8))
plt.title("Linear Regressor")
plt.plot_date(y= regressor.predict(X_test.values),x= predictions_dates1, linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y = y_test.values,x= predictions_dates1,linestyle = '-',color = 'blue',label = 'Actual')
#plt.plot_date(y = regressor.predict(x_valid.values),x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
plt.ylabel('amount')
plt.xlabel('date')
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(20,8))
plt.title("Linear Regressor")
plt.plot_date(y = regressor.predict(x_valid.values),x= predictions_dates,linestyle = '-',color = '#ff9999',label = 'Actual')
plt.plot_date(y = y_valid.values,x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
#plt.plot_date(y = regressor.predict(x_valid.values),x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
plt.ylabel('amount')
plt.xlabel('date')
plt.legend(loc="best")
plt.show()


plt.figure(figsize=(20,8))
plt.title("Random Forest Regressor")
plt.plot_date(y= rfe.predict(X_test2),x= predictions_dates, linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y = y_test2.values,x= predictions_dates,linestyle = '-',color = 'blue',label = 'Actual')
plt.ylabel('amount')
plt.xlabel('date')
plt.legend(loc="best")
plt.show()


