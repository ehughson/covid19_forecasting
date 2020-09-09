import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from datetime import datetime, date, timedelta
from dateutil import parser
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

#plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

#reference: https://machinelearningmastery.com/make-sample-forecasts-arima-python/
#reference: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
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

#################### PREDICTING HOSPITAL BED FORECASTING ###########################
df = pd.read_csv("covid19-NatEst.csv", header = 0, index_col = 0)
df = df[['State name', 'Day for which estimate is made','Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate']]
df = df.rename(columns={"Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate" : "bed amount"})
df = df.rename(columns={'Day for which estimate is made' : "date"})
df = df[df['State name'] != 'United States']
df = df[df['State name'] != 'District of Columbia']
df['date'] = df['date'].apply(date_to_datetime)
df = df.sort_values('date')
df = df.groupby(['date']).sum().reset_index()
df =df.set_index('date')
#print(df.head(20))
split = len(df) - 7
dataset, validation = df[0:split], df[split:]
#print(validation.values)
#print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		print(dataset[i])
		value = dataset[i] - dataset[i - interval].astype(int)
		diff.append(value)
	return np.array(diff)

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


######## gather dates to use for plotting ###############
predictions_dates = []
start = pd.to_datetime('2020-07-01')
pred = []
for i in range(7):
	date = start + timedelta(days = 1)
	#print(date)
	predictions_dates.append(date)
	start = date

all_dates = []
start = pd.to_datetime('2020-04-01')
for i in range(len(df)-7):
	date = start + timedelta(days = 1)
	#print(date)
	all_dates.append(date)
	start = date

# load dataset
series = pd.read_csv('dataset.csv')
#test = pd.read_csv('validation.csv')
#print(series)
#print(series['bed amount'].dtype())
# seasonal difference
X = series.values
#y = df['bed amount'].values
#X = df['date'].values
#print(X)
#days_in_year = 365
differenced = difference(X)
#differenced = difference(differenced)

### make model and fit it ###############
model = ARIMA(X, order=(5,2,1))
model_fit = model.fit(trend = 'c', full_output = True, disp=0)
# print summary of fit model
print(model_fit.summary())


#reference: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
##### plot the data ############
model_fit.plot_predict(dynamic = False)
plt.show()
forecast, se, conf = model_fit.forecast(steps = 7, alpha = 0.05)
fc_data = pd.Series(forecast, index = validation.index)
lower_bound = pd.Series(conf[:, 0], index = validation.index)
upper_bound = pd.Series(conf[:, 1], index = validation.index)

plt.figure(figsize=(10,5))
plt.xlabel("dates")
plt.ylabel("amount")
plt.plot(all_dates, dataset.values, label='training')
plt.plot(predictions_dates,validation.values, label='actual')
plt.plot(predictions_dates,fc_data.values, label='forecast')
plt.fill_between(lower_bound.index, lower_bound, upper_bound, 
                 color='k', alpha=.15)
plt.title('Forecast vs. Actuals For Bed Occupancy')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def f_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) 
    print("MAPE is: " + str(mape))
    rmse = np.sqrt(np.mean((forecast - actual)**2))
    print("RMSE is:" + str(rmse))

f_accuracy(forecast, validation.values)


#####plot forecast data
forecast, se, conf = model_fit.forecast(steps = 50, alpha = 0.05)
index = np.arange(len(dataset), len(dataset)+50)

fc_data = pd.Series(forecast, index=index)
lower_bound = pd.Series(conf[:, 0], index=index)
upper_bound = pd.Series(conf[:, 1], index=index)

plt.figure(figsize=(10,5))
plt.xlabel("steps")
plt.ylabel("amount")
plt.plot(df['bed amount'].values)
plt.plot(fc_data, color='darkgreen')
plt.fill_between(lower_bound.index, lower_bound, upper_bound,  alpha=.15)
plt.title("Final Forecast of Bed Occupancy")
plt.show()


#################### PREDICTING CASES FORECASTING ###########################
def get_differences(df):
	df['next_case'] = df['cases']
	df['next_case'] = df['cases'].shift(periods = -1)
	df['difference'] = df['next_case'] - df['cases']
	#print(df.head())
	return df
 

 #### gather data for covid - 19 cases ###########

df = pd.read_csv("us-states.csv", header = 0)
#print(df.head())
df = df[['date','state','cases']]
df = df[df['state'] != 'District of Columbia']
df['date'] = df['date'].apply(date_to_datetime)
df = df.sort_values('date')
df = df.groupby(['date']).sum().reset_index()
df =df.set_index('date')
##### difference the values so that we can get cases per day and not cumulative cases
df = get_differences(df)
df= df.drop(columns = ['cases','next_case'])
df = df.dropna()
#k = df.loc[:,'2020-04-05':]
#df_beds = k.values.tolist()[0]
#print(df_beds)
#print(df.tail(30))

split = len(df) - 7
#print(split_point)

#####train test split ###########
dataset, validation = df[0:split], df[split:]
#print(dataset.head())
#print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('casedataset.csv', index=False)
validation.to_csv('casevalidation.csv', index=False) ## to test dataset

#print(validation.tail())
predictions_dates = []
start = pd.to_datetime('2020-07-18')
pred = []
day = 1

for i in range(7):
	date = start + timedelta(days = 1)
	#print(date)
	predictions_dates.append(date)
	start = date

all_dates = []
start = pd.to_datetime('2020-01-21')
for i in range(len(df)-7):
	date = start + timedelta(days = 1)
	#print(date)
	all_dates.append(date)
	start = date

#print(all_dates[-1])

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		print(dataset[i])
		value = dataset[i] - dataset[i - interval].astype(int)
		diff.append(value)
	return np.array(diff)


# load dataset
series = pd.read_csv('casedataset.csv')
X = series.values
#print(X)

#### make model and fit it #############
model = ARIMA(X, order=(5,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
model_fit.plot_predict(dynamic = False)
#plt.show()

forecast, se, conf = model_fit.forecast(steps = 7, alpha = 0.05)
fc_data = pd.Series(forecast, index = validation.index)
lower_bound = pd.Series(conf[:, 0], index = validation.index)
upper_bound = pd.Series(conf[:, 1], index = validation.index)

# Plot
plt.figure(figsize=(10,5))
plt.xlabel("dates")
plt.ylabel("amount")
plt.plot(all_dates, dataset.values, label='training')
plt.plot(predictions_dates,validation.values, label='actual')
plt.plot(predictions_dates, fc_data.values, label='forecast')
plt.fill_between(lower_bound.index, lower_bound, upper_bound, alpha=.15)
plt.title('Forecast vs Actuals for Covid-19 Cases')
plt.legend(loc='upper left', fontsize=8)
plt.show()


#### get accuracy scores ############
def f_accuracy(forecast, actual):
    mape = np.mean((forecast - actual)/(actual))  # MAPE
    print("MAPE is: " + str(mape))
    rmse = np.sqrt(np.mean((forecast - actual)**2))
    print("RMSE is:" + str(rmse))
  
f_accuracy(forecast, validation.values)

forecast, se, conf = model_fit.forecast(steps = 50, alpha = 0.05)
index = np.arange(len(dataset), len(dataset)+50)
fc_data = pd.Series(forecast, index=index)
lower_bound = pd.Series(conf[:, 0], index=index)
upper_bound = pd.Series(conf[:, 1], index=index)

predictions_dates = []
start_date = pd.to_datetime('2020-07-18')
pred = []
day = 1

for i in range(50):
	date = start_date + timedelta(days = 1)
	#print(date)
	predictions_dates.append(date)
	start_date = date

all_dates = []
start_date = pd.to_datetime('2020-01-21')
for i in range(len(dataset)):
	date = start_date + timedelta(days = 1)
	#print(date)
	all_dates.append(date)
	start_date = date

### plot data ########
plt.figure(figsize=(10,5))
plt.xlabel("steps")
plt.ylabel("amount")
plt.plot(df['difference'].values)
plt.plot(fc_data, color='darkgreen')
plt.fill_between(lower_bound.index, lower_bound, upper_bound, alpha=.15)
plt.title("Final Forecast of Covid-19 Cases")
plt.show()

