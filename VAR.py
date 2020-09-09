import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from dateutil import parser
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
#from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import acf

#references: https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
#references: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
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

def get_differences(df):
	df['next_case'] = df['cases']
	df['next_case'] = df['cases'].shift(periods = -1)
	df['difference'] = df['next_case'] - df['cases']
	#print(df.head())
	return df

########## read cases csv ######################

df = pd.read_csv("us-states.csv", header = 0)
#df = df[['date','state','cases', 'deaths']]
df = df[['date','state','cases']]
df = df[df['state'] != 'District of Columbia']
df['date'] = df['date'].apply(date_to_datetime)
#df['population'] = df['state'].apply(add_population)
#df['population_weight'] = df['cases'] / df['population'] * 100

######## Read hospital bed csv ################
### TRANSFORM DATA BY GROUPBY AND MERGINING DATA ON DATE AND STATE VALUES #########
df2 = pd.read_csv("covid19-NatEst.csv", header = 0, index_col = 0)
df2 = df2[['State name', 'Day for which estimate is made','Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate']]
df2 = df2.rename(columns={"Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate" : "bed amount"})
df2 = df2.rename(columns={'Day for which estimate is made' : "date"})
df2= df2[df2['State name'] != 'United States']
df2 = df2[df2['State name'] != 'District of Columbia']
df2['dateTime'] = df2['date'].apply(date_to_datetime)
df2 = df2.drop(columns = 'date')
new_df = df2.merge(df, left_on=['dateTime','State name'], right_on = ['date','state'], how='left')
new_df = new_df.drop(columns = 'dateTime')
new_df = new_df.sort_values('date')
#new_df = new_df.groupby("date").sum().reset_index()
#print(new_df)
data = new_df.drop(columns = 'date', axis = 1)
data.index = new_df.date

data = data.drop(columns = ['state','State name'])
data.dropna(inplace=True)


####################### VAR model attempt #2 ##########################


##### run grangers causation matrix  on data to assess if attributes have causality 
def grangers_causation(data):    
    for c in data.columns:
        for r in data.columns:
            result = grangercausalitytests(data[[r, c]], maxlag=2, verbose=False)
            p_values = np.min([result[i+1][0]['ssr_chi2test'][1] for i in range(2)])
            #p_value = np.min(p_values)
            if r != c:
            	print("p-value for " + r + ", " + c + " is: " + str(p_values))
            	if p_values < 0.05:
            		print("Statistically significant!!")

df = new_df.drop(columns = ['state','State name'])
df = df.groupby("date").sum().reset_index()
df = df.set_index('date')
grangers_causation(df) 
#granger_test = grangercausalitytests(df, maxlag=2, verbose=True)
#print(granger_test)

##### run adfuller test on data to assess if stationary #################
def adfuller_test(df):
    r = adfuller(df, autolag='AIC')
    p_value = round(r[1], 4)
    if p_value <= 0.05:
        print("p-value = " +  str(p_value) + ": Can reject the Null Hypothesis because it seems stationary")
    else:
        print("p-value = " +  str(p_value) + ": Cannot reject the Null Hypothesis because it seems non-stationary")

#nobs = 4
df_train = df[:int(0.9*(len(df)))]
df_test = df[int(0.9*(len(df))):]
print(df_train.head())
# Check size
print(df_train.shape)  
print(df_test.shape) 

###### difference the dataset to make it stationary #############
df_differenced = df_train.diff().dropna()
df_differenced = df_differenced.diff().dropna()

###### assess adfuller test ##############
adfuller_test(df_differenced['bed amount'])
adfuller_test(df_differenced['cases'])
    

###create and fit model 
model = VAR(df_differenced)
model_fit = model.fit(maxlags=7, ic='aic')
print(model_fit.summary())

length = (len(df_test))
columns = ['bed amount_differenced2', 'cases_differenced2']
forecast_mf = model_fit.forecast(y=df_differenced.values[-model_fit.k_ar:], steps=length)
forecast_df = pd.DataFrame(forecast_mf, index=df.index[-length:], columns=columns)
#print(forecast_df)


#references: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/ 
### need to invert data back to initial form before seeing outcome of data ##########
def invert_transformation(df_train, df_forecast, second_diff=False):
    df = df_forecast
    for col in df_train.columns:        
        if second_diff:
            df[str(col)+'_differenced'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df[str(col)+'_differenced2'].cumsum()
        df[str(col)+'_prediction'] = df_train[col].iloc[-1] + df[str(col)+'_differenced'].cumsum()
    return df

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

df_results = invert_transformation(df_train, forecast_df, second_diff=True)     
#print(df_results)


###### plot the results of VAR ###########
fig, axes = plt.subplots(ncols=2,figsize=(10,5))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_prediction'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-length:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")

plt.tight_layout();
plt.show()

def f_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) 
    print("MAPE is: " + str(mape))
    rmse = np.sqrt(np.mean((forecast - actual)**2))
    print("RMSE is:" + str(rmse))

 
######## PRINT ACCURACY VALUES #####################
print('Accuracy of: bed amount')
accuracy_prod = f_accuracy(df_results['bed amount_prediction'].values, df_test['bed amount'])
print('\nAccuracy of: cases')
accuracy_prod = f_accuracy(df_results['cases_prediction'].values, df_test['cases'])

model_fit.plot_forecast(50)
plt.show()



