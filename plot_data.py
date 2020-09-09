import pandas as pd
#import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objects as go
import re
from dateutil import parser
from datetime import datetime, date
import time


#references: https://plotly.com/python/choropleth-maps/
#references: https://plotly.com/python/pie-charts/
def add_code(row):
    #print(row['province'])
    if row['state'] == 'Alabama':
        value = 'AL'
        return value
    elif row['state'] == 'Alaska':
        value = 'AK'
        return value
    elif row['state'] == 'Arizona':
        value = 'AZ'
        return value
    elif row['state'] == 'Arkansas':
        value = 'AR'
        return value
    elif row['state'] == 'California':
        value = 'CA'
        return value
    elif row['state'] == 'Colorado':
        value = 'CO'
        return value
    elif row['state'] == 'Connecticut':
        value = 'CT'
        return value
    elif row['state'] == 'Delaware':
        value = 'DE'
        return value
    elif row['state'] == 'Florida':
        value = 'FL'
        return value
    elif row['state'] == 'Georgia':
        value = 'GA'
        return value
    elif row['state'] == 'Hawaii':
        value = 'HI'
        return value
    elif row['state'] == 'Idaho':
        value = 'ID'
        return value
    elif row['state'] == 'Illinois':
        value = 'IL'
        return value
    elif row['state'] == 'Indiana':
        value = 'IN'
        return value
    elif row['state'] == 'Iowa':
        value = 'IA'
        return value
    elif row['state'] == 'Kansas':
        value = 'KS'
        return value
    elif row['state'] == 'Kentucky':
        value = 'KY'
        return value
    elif row['state'] == 'Louisiana':
        value = 'LA'
        return value
    elif row['state'] == 'Maine':
        value = 'ME'
        return value
    elif row['state'] == 'Maryland':
        value = 'MD'
        return value
    elif row['state'] == 'Massachusetts':
        value = 'MA'
        return value
    elif row['state'] == 'Michigan':
        value = 'MI'
        return value
    elif row['state'] == 'Minnesota':
        value = 'MN'
        return value
    elif row['state'] == 'Mississippi':
        value = 'MS'
        return value
    elif row['state'] == 'Missouri':
        value = 'MO'
        return value
    elif row['state'] == 'Montana':
        value = 'MT'
        return value
    elif row['state'] == 'Nebraska':
        value = 'NE'
        return value
    elif row['state'] == 'Nevada':
        value = 'NV'
        return value
    elif row['state'] == 'New Hampshire':
        value = 'NH'
        return value
    elif row['state'] == 'New Jersey':
        value = 'NJ'
        return value
    elif row['state'] == 'New Mexico':
        value = 'NM'
        return value
    elif row['state'] == 'New York':
        value = 'NY'
        return value
    elif row['state'] == 'North Carolina':
        value = 'NC'
        return value
    elif row['state'] == 'North Dakota':
        value = 'ND'
        return value
    elif row['state'] == 'Ohio':
        value = 'OH'
        return value
    elif row['state'] == 'Oklahoma':
        value = 'OK'
        return value
    elif row['state'] == 'Oregon':
        value = 'OR'
        return value
    elif row['state'] == 'Pennsylvania':
        value = 'PA'
        return value
    elif row['state'] == 'Rhode Island':
        value = 'RI'
        return value
    elif row['state'] == 'South Carolina':
        value = 'SC'
        return value
    elif row['state'] == 'South Dakota':
        value = 'SD'
        return value
    elif row['state'] == 'Tennessee':
        value = 'TN'
        return value
    elif row['state'] == 'Texas':
        value = 'TX'
        return value
    elif row['state'] == 'Utah':
        value = 'UT'
        return value
    elif row['state'] == 'Vermont':
        value = 'VT'
        return value
    elif row['state'] == 'Virginia':
        value = 'VA'
        return value
    elif row['state'] == 'Washington':
        value = 'WA'
        return value
    elif row['state'] == 'West Virginia':
        value = 'WV'
        return value
    elif row['state'] == 'Wisconsin':
        value = 'WI'
        return value
    elif row['state'] == 'Wyoming':
        value = 'WY'
        return value
    else:
        return "unknown"
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
    value = '%04i-%02i' % (value.year, value.month)
    return value

def date_to_month(d):
    # You may need to modify this function, depending on your data types.
    #month = d.month
    if len(d) > 10:
    	m = re.search(r'(?<=-)\w+', d)
    	d = m.group(0)
    d = d[3:]
    #print(d)
    #print(month)
    return d


def get_differences_bed(df):
	df['dateTime']  = pd.to_datetime(df['date'])
	df = df.groupby(['dateTime'])['cases'].sum().reset_index()
	df['next_case'] = df['cases']
	df['next_case'] = df['cases'].shift(periods = -1)
	df['difference'] = df['next_case'] - df['cases']
	#print(df.head())
	plt.bar(df['dateTime'], df['difference'], color='g')
	plt.title('Covid-19 cases per day in the United States')
	plt.show()
	return df


###### EXTRACT AND PREPARE CDC DATASET ###########################
df = pd.read_csv("covid19-NatEst.csv")

#for columns in df:
#	print(columns)

df2 = df[['State name','Day for which estimate is made','Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate']]
df2 = df2[df2['State name'] != 'United States']
#df2 = df2[df2['State name'] == 'California']
df2 = df2.dropna()
df2['dateTime'] = df2['Day for which estimate is made'].apply(date_to_datetime)
df2['state'] = df2['State name']
df2['code'] = df2.apply(lambda row: add_code(row), axis=1)

df6 = df2[(df2['state'] == 'California') | (df2['state'] == 'Florida') | (df2['state'] == 'New York') | (df2['state'] == 'Texas')]
#fig, ax = plt.subplots(figsize=(10,5))

##### EXPLORE MOST EFFECTED STATES ###########
df6 = df6.groupby(['dateTime', 'State name'])['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'].sum().unstack().plot()
plt.title("Bed Occupancy Overtime in States with High Bed Occupancy")
plt.ylabel('amount')
plt.xlabel('date')
plt.show()

#df21 = df2.groupby(['dateTime']).agg({'Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate': 'sum'}).reset_index()
#plt.bar(df21['dateTime'], df21['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'], color='g')
#plt.show()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


df6 = df2.groupby(['code']).agg({'Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate': 'max'}).reset_index()
value = df6['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'].sum()
#print(value)
df7= df6
df7['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'] = df7['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate']/value
#print(df7.head())
#perstate = df11[df11['code'] != '']['cases'].max().to_dict()
perstate1 =pd.Series(df6['Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate'].values,index=df6.code).to_dict()
#print(perstate1)


############CREATE PIE CHART ##########################
fig = px.pie(df7, values='Number of patients in an inpatient care location who have suspected or confirmed COVID-19,  estimate', names='code', title="Pie Chart of Beds Occupied by Covid-19 Patients for Each State")
#fig.title("Pie Chart of Beds Occupied by Covid-19 Patients for Each State")
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()

########### CREATE HEAT MAP #########################
data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'reds',
        locations = list(perstate1.keys()),
        locationmode = 'USA-states',
        #text = list(perstate.values()),
        z = list(perstate1.values()),
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            ),
        )]

layout = dict(
         title = 'Bed Occupancy by states',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

fig= dict(data = data, layout = layout)
iplot(fig)


######### CREATE BAR GRAPH ####################
def get_differences(df):
	df['dateTime']  = pd.to_datetime(df['date'])
	df = df.groupby(['dateTime'])['cases'].sum().reset_index()
	df['next_case'] = df['cases']
	df['next_case'] = df['cases'].shift(periods = -1)
	df['difference'] = df['next_case'] - df['cases']
	print(df.head())
	plt.bar(df['dateTime'], df['difference'], color='g')
	plt.title('Covid-19 cases per day in the United States')
	plt.xlabel('date')
	plt.ylabel('amount')
	plt.show()
	return df


###### EXTRACT AND PREPARE NYT DATASET ###########################

df10 = pd.read_csv("us-states.csv")
df10['dateTime'] = df10['date'].apply(date_to_datetime)
print(df10.head())
df11 = df10.groupby(['dateTime','state']).agg({'cases': 'max'}).reset_index()
print(df11)
#df11 = df11[df11['state']=='California']
df_difference = get_differences(df10)
#plt.plot(df11['dateTime'], df11['cases'], color='g')
#plt.show()

###### EXPLORE TOP STATES ######################### 
df6 = df10[(df10['state'] == 'California') | (df10['state'] == 'Florida') | (df10['state'] == 'New York') | (df10['state'] == 'Texas')]
fig, ax = plt.subplots(figsize=(10,5))
df6 = df6.groupby(['dateTime', 'state'])['cases'].sum().unstack().plot()
plt.title("Covid-19 Cases Overtime in States with High Case Numbers")
plt.ylabel('amount')
plt.xlabel('date')
plt.show()


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df10['code'] = df10.apply(lambda row: add_code(row), axis=1)
df12 = df10.groupby(['code']).agg({'cases': 'max'}).reset_index()
value = df12['cases'].sum()
#print(value)
df13 = df12
df13['cases'] = df13['cases']/value
#print(df13)
#perstate = df11[df11['code'] != '']['cases'].max().to_dict()
perstate =pd.Series(df12.cases.values,index=df12.code).to_dict()
#print(perstate)

################## CREATE PIE CHART #############################
fig = px.pie(df13, values='cases', names='code', title ="Pie Chart of Covid-19 Cases for Each State")
#fig.title("Pie Chart of Covid-19 Cases for Each State")
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()

################# CREATE HEAT MAP ################################

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Blues',
        locations = list(perstate.keys()),
        locationmode = 'USA-states',
        #text = list(perstate.values()),
        z = list(perstate.values()),
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            ),
        )]

layout = dict(
         title = 'Covid cases by states',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

fig= dict(data = data, layout = layout)
iplot(fig)



