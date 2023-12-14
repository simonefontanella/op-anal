import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(x, y):
    plt.plot(x, y)
    plt.show()


# load data
data = pd.read_csv("rimini_climate.txt", sep=",")
# data exploration
print("----------------\nData exploration\n----------------")
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print("----------------")


data = data.drop(columns=["min_temp", "max_temp"])

print("----------------\nData after drop min_temp, max_temp\n----------------")
print(data.head(5))
print("----------------")
"""
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data['avg_temp'][:200000], lags=20)  # Lags indica quanti ritardi visualizzare
plt.title('Autocorrelogramma della Serie Temporale')
plt.xlabel('Ritardi Temporali')
plt.ylabel('Valore di Autocorrelazione')
plt.show()
"""


# average of avg_temp group by day (21915)

group_df = data.groupby("day")["avg_temp"].mean().reset_index(name ='avg_temp')
group_df["day"] = pd.to_datetime(group_df["day"], format="%Y%m%d")
group_df.reset_index(drop=True, inplace=True)
group_df = group_df.set_index("day")

# average of temp group by month
group_df["month"] = group_df.index.month
group_df["year"] = group_df.index.year
pivot_month = pd.pivot_table(group_df, values = "avg_temp", index="month", columns="year", aggfunc="mean")
print(pivot_month)
pivot_month.plot()
plt.show()


print("----------------\nGroup by day\n----------------")
print(group_df.head(5))
print(group_df.describe())
print(group_df.tail(20))
print("----------------")
group_df = data.groupby("day")["avg_temp"].mean().reset_index(name ='avg_temp')


"""
fig, axes = plt.subplots(2)
axes[0].plot(group_df["avg_temp"][:2000])


diff_values = group_df["avg_temp"].diff().dropna()

axes[1].plot(diff_values[:2000])
plt.show()
fig, axes = plt.subplots(2)
diff_values = group_df["avg_temp"].diff().dropna()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf( group_df["avg_temp"], lags=90, ax = axes[0])  # Lags indica quanti ritardi visualizzare
plot_acf( diff_values, lags=90, ax = axes[1])  # Lags indica quanti ritardi visualizzare

plt.title('Autocorrelogramma della Serie Temporale')
plt.xlabel('Ritardi Temporali')
plt.ylabel('Valore di Autocorrelazione')
plt.show()
"""