code files & references:
RUN:  "python3 <file name>"  in terminal

1. plot_data.py -- this is where the initial visuals are produced during exploratory analsys. Include Heat map, pie chart, and bar/line graphs
References:
 	1. references: https://plotly.com/python/choropleth-maps/
	2. references: https://plotly.com/python/pie-charts/

2. ARIMA.py -- this is where the ARIMA model is 
References:
	1. reference: https://machinelearningmastery.com/make-sample-forecasts-arima-python/
	2. reference: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

3. regressions.py -- this is where the regression models are (i.e., linear regression and random forest regressor)
References:
	1. reference: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
	2. reference: https://www.kaggle.com/pbizil/random-forest-regression-for-time-series-predict

4. VAR.py -- this is where the VAR model is
References:
	1. references: https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
	2. references: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

# Datasets
> Repo contains two datasets: (1) New York Time’s Covid-19 GitHub (called “us_states.csv”) dataset which addresses confirmed Covid-19 cases in the United States and (2) the other dataset (called “covid19-NatEst.csv) is from the Center for Disease Control and
Prevention (CDC) about hospital bed occupancy.
	
## New York Times Dataset
> The New York Times is the cumulative number of confirmed Covid-19 cases each day for each state in the United States. It is updated every day and provides up to date information from January 21st to July 25th, 2020. Historical data was used as it contained a complete count for a given day, while live count may have partial counts. Data is available on a county level, state level and country level. The state level data was used because it drilled down on the statewide comparison.
> The data has 5 columns: date, state, fips, cases, and deaths. For the current analysis, date, state, and cases are used. The date attribute is the date the data was collected, state attribute was which Covid-19 Cases and Inpatient Hospital Bed Capacity
> state the data was collected for, and the cases attribute is the number of cumulative cases that day for a given state. The data is collected by journalists who extract information from news conferences, data releases, and public officials. The creator(s) of the dataset also noted that government official constantly change case numbers and provide inconsistent information. Confirmed patients are counted by where they are seeking treatment and are counted only if they have been confirmed a laboratory test and is reported by some level of government agency.
	
## CDC Dataset
> The CDC dataset can be found on the CDC’s website under current hospital capacity estimates. This page is dedicated to Covid-19 related data and focuses on the current inpatient and ICU bed occupancy estimates. The data is submitted by hospitals in the United States to the NHSN COVID-19 Module and is weighted and imputed to account for non-responses and missing data.
> Collection occurred between April 4th to July 7th, 2020. The data comes with the estimated number of beds being occupied by Covid-19 patients in both inpatient and ICU areas of hospitals, confidence intervals, and the amount of inpatient and ICU beds occupied by all patients. 
> This repo focuses on only inpatient beds not ICU beds. 
>> The attributes used for analysis:
	- ‘Day for which estimate is made’ (renamed date)
	- ‘Number of patients in an inpatient care location who have suspected or confirmed COVID-19, percent estimate (percent ofinpatient beds)’ 
	- 'Number of patients in an inpatient care location who have suspected or confirmed COVID-19, estimate' (renamed bed amount). 
	The current data for hospital beds is only reported up to July 14th, 2020 and has not been updated since. 
