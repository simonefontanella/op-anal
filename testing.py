import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(x, y):
    plt.plot(x, y)
    plt.show()


# load data
data = pd.read_csv("../dataset/climate.txt", sep=",")
# data exploration
print("----------------\nData exploration\n----------------")
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print("----------------")

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data[' temp_avg'][:20000], lags=20)  # Lags indica quanti ritardi visualizzare
plt.title('Autocorrelogramma della Serie Temporale')
plt.xlabel('Ritardi Temporali')
plt.ylabel('Valore di Autocorrelazione')
plt.show()

exit(0)

plt.plot(data['close'],label='close')
plt.plot(data['open'],label='open')
plt.plot(data['high'],label='high')
plt.plot(data['low'],label='low')
plt.legend()

plt.show()

# Rimuovi il trend tramite differenziazione
serie_temporale_senza_trend = np.diff(data['close'])

# Visualizza la serie temporale senza trend
plt.figure(figsize=(10, 5))
plt.plot(serie_temporale_senza_trend, label='Serie Temporale senza Trend')
plt.title('Serie Temporale Senza Trend (Differenziazione)')
plt.legend()
plt.show()

# annual_rate = data.groupby("Year")["Rate"].mean()

# # Plotting
# plt.plot(
#     annual_rate.index, annual_rate.values, label="Annual mean US unemployment rate"
# )
# plt.title("Annual mean US unemployment rate")
# plt.xlabel("Year")
# plt.ylabel("Unemployment Rate")
# plt.legend()
# # plt.show()

# df = data.drop(columns=["Month", "State", "County"])
# print(df)


# def difference(data, interval=1):
#     return [data[i] - data[i - interval] for i in range(interval, len(data))]


# diff_values = df["Rate"].diff().dropna()
# # plt.plot(diff_values)
# # plt.plot(df["Rate"])
# # plt.show()


exit(0)