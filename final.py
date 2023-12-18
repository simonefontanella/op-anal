import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import math
from statsmodels.tsa.seasonal import seasonal_decompose


#Load data from csv
data = pd.read_csv("rimini_climate.txt", sep=",")

#Visualize data
print("----------------\nData exploration\n----------------")
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print("----------------")

# Preprocessing data
data = data.drop(columns=["min_temp", "max_temp"])

data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
data.reset_index(drop=True, inplace=True)
data = data.set_index("date")

print("----------------\nData after drop min_temp, max_temp\n----------------")
print(data.head(5))
print("----")


# raggruppiamo le temperature ottenute in zone diverse di Rimini per lo stesso giorno aggregandole con la media
data["month"] = data.index.month
data["year"] = data.index.year


# aggreghiamo le informazioni, notando un picco di temperatura nei mesi estivi come ci si poteva aspettare
pivot_month = pd.pivot_table(data, values = "temp", index="month", columns="year", aggfunc="mean")
print(pivot_month)
pivot_month.plot()
plt.show()

#facendo la media mobile su 10 anni notiamo un trend lievemente crescente per la temperatura media 
year_avg = pd.pivot_table(data, values='temp', index='year', aggfunc='mean')
year_avg['10 Years MA'] = year_avg['temp'].rolling(10).mean()
year_avg[['temp','10 Years MA']].plot(figsize=(20,6))
plt.title('Avg temperatures in Rimini aggregate per years')
plt.xlabel('Months')
plt.ylabel('Temperature')
plt.xticks([x for x in range(1961,2021,2)])
plt.show()


# prima di procedere con il training dei modelli aggreghiamo i dati per mese tramite media
group_df = data.groupby(["year", "month"])["temp"].mean().reset_index(name ='temp')
group_df
print('Aggregate data by year and month')
print(group_df)

# search for outliers: STL decomposition
# study the residuals and seasonality

result = seasonal_decompose(group_df['temp'][:48], model='additive', period=12, extrapolate_trend='freq')
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
std = resid.std()
plt.plot(resid, "o", label="datapoints")
outliers = pd.Series(resid)
outliers = pd.concat([outliers[outliers < (-1.5 * std)], outliers[outliers > (1.5 * std)]])
plt.plot(outliers, "*", color='violet', label="outliers")
plt.hlines(0, 0, len(resid))
plt.hlines(1.5 * std, 0, len(resid), color="red", label="std limits")
plt.hlines(-1.5 * std, 0, len(resid), color="red")
plt.title('STL decomposition')
plt.legend()
plt.show()
plt.subplot(2, 1, 1)
plt.plot(observed, label='observed')
plt.plot(trend, label='trend')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(seasonal, label='seasonal')
plt.plot(resid, label='residual')
plt.legend()
plt.show()

#controllo della stazionarita' della serie
def check_stationarity(y, lags_plots=48, figsize=(22,8)):
    "Use Series as parameter"
    
    # Creating plots of the DF
    y = pd.Series(y)
    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    y.plot(ax=ax1, figsize=figsize)
    ax1.set_title('Rimini Temperatures')
    plot_acf(x= y, lags=lags_plots, zero=False, ax=ax2)
    
    plt.tight_layout()
    
    print('Results of Dickey-Fuller Test:')
    adfinput = adfuller(y)
    adftest = pd.Series(adfinput[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    adftest = round(adftest,4)
    
    for key, value in adfinput[4].items():
        adftest[f"Critical Value {key}"] = value.round(4)
        
    print(adftest)
    
    if adftest[0].round(2) < adftest[5].round(2):
        print('\nThe Test Statistics is lower than the Critical Value of 5%.\nThe serie seems to be stationary')
    else:
        print("\nThe Test Statistics is higher than the Critical Value of 5%.\nThe serie isn't stationary")