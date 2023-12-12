import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(x, y):
    plt.plot(x, y)
    plt.show()


# load data
data = pd.read_csv("..\\dataset\\output.csv", sep=",")
# data exploration
print("----------------\nData exploration\n----------------")
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print("----------------")

#####################
# preprocessing
#####################
# Annual.rate <- sapply(split(unemp.data$Rate, unemp.data$Year), mean)
# Year = c(1990:2016)
# plot(Year, Annual.rate,  main = "Annual mean US unemployment rate"); lines(Year, Annual.rate)

# Assuming unemp.data is a DataFrame with columns 'Year' and 'Rate'
# Sample data creation for demonstration purposes

# Calculate annual mean unemployment rate
annual_rate = data.groupby("Year")["Rate"].mean()

# Plotting
plt.plot(
    annual_rate.index, annual_rate.values, label="Annual mean US unemployment rate"
)
plt.title("Annual mean US unemployment rate")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.legend()
# plt.show()

df = data.drop(columns=["Month", "State", "County"])
print(df)


def difference(data, interval=1):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]


diff_values = df["Rate"].diff().dropna()
# plt.plot(diff_values)
plt.plot(df["Rate"])
plt.show()

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Creazione di una serie temporale casuale
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31')
values = np.random.randint(0, 100, size=len(dates))

# Creazione di un DataFrame Pandas con le date e i valori
data = pd.DataFrame({'Date': dates, 'Value': values})
data.set_index('Date', inplace=True)

# Grafico della serie temporale
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Value'], label='Serie Temporale', color='blue')
plt.title('Serie Temporale')
plt.xlabel('Data')
plt.ylabel('Valore')
plt.legend()
plt.grid(True)
plt.show()


"""
