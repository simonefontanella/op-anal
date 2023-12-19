import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from dm_test import dm_test
import pmdarima as pm
from statsmodels.tools.eval_measures import mse, rmse

# Load data from csv
data = pd.read_csv("rimini_climate.txt", sep=",")

# Visualize data
print("----------------\nData exploration\n----------------")
print(data.info())
print(data.describe())
print(data.shape)
print(data.head(5))
print("----------------")

# Preprocessing dati
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
pivot_month = pd.pivot_table(
    data, values="temp", index="month", columns="year", aggfunc="mean"
)
print(pivot_month)
pivot_month.plot()
plt.show()

# facendo la media mobile su 10 anni notiamo un trend lievemente crescente per la temperatura media
year_avg = pd.pivot_table(data, values="temp", index="year", aggfunc="mean")
year_avg["10 Years MA"] = year_avg["temp"].rolling(10).mean()
year_avg[["temp", "10 Years MA"]].plot(figsize=(20, 6))
plt.title("Avg temperatures in Rimini aggregate per years")
plt.xlabel("Months")
plt.ylabel("Temperature")
plt.xticks([x for x in range(1961, 2021, 2)])
plt.show()


# prima di procedere con il training dei modelli aggreghiamo i dati per mese tramite media
group_df = data.groupby(["year", "month"])["temp"].mean().reset_index(name="temp")
group_df
print("Aggregate data by year and month")
print(group_df)

# search for outliers: STL decomposition
# study the residuals and seasonality

result = seasonal_decompose(
    group_df["temp"][:48], model="additive", period=12, extrapolate_trend="freq"
)
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
std = resid.std()
plt.plot(resid, "o", label="datapoints")
outliers = pd.Series(resid)
outliers = pd.concat(
    [outliers[outliers < (-1.5 * std)], outliers[outliers > (1.5 * std)]]
)
plt.plot(outliers, "*", color="violet", label="outliers")
plt.hlines(0, 0, len(resid))
plt.hlines(1.5 * std, 0, len(resid), color="red", label="std limits")
plt.hlines(-1.5 * std, 0, len(resid), color="red")
plt.title("STL decomposition")
plt.legend()
plt.show()
plt.subplot(2, 1, 1)
plt.plot(observed, label="observed")
plt.plot(trend, label="trend")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(seasonal, label="seasonal")
plt.plot(resid, label="residual")
plt.legend()
plt.show()


# controllo della stazionarita' della serie
def check_stationarity(y, lags_plots=48, figsize=(22, 8)):
    "Use Series as parameter"

    y = pd.Series(y)
    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    y.plot(ax=ax1, figsize=figsize)
    ax1.set_title("Rimini Temperatures")
    plot_acf(x=y, lags=lags_plots, zero=False, ax=ax2)

    plt.tight_layout()

    print("Results of Dickey-Fuller Test:")
    adfinput = adfuller(y)
    adftest = pd.Series(
        adfinput[0:4],
        index=["Test Statistic", "p-value", "Lags Used", "Number of Observations Used"],
    )
    adftest = round(adftest, 4)

    for key, value in adfinput[4].items():
        adftest[f"Critical Value {key}"] = value.round(4)

    print(adftest)

    if adftest[0].round(2) < adftest[5].round(2):
        print(
            "\nThe Test Statistics is lower than the Critical Value of 5%.\nThe serie seems to be stationary"
        )
    else:
        print(
            "\nThe Test Statistics is higher than the Critical Value of 5%.\nThe serie isn't stationary"
        )
    plt.show()

# Per creare una previsione di serie temporali, la serie deve essere stazionaria
# Un modo per verificare se la serie è stazionaria è utilizzare la funzione adfuller, se il valore P è inferiore al 5% (numero abituale utilizzato per questo tipo di studio) la serie è stazionaria
# Ipotesi nulla (H0): La serie temporale ha una radice unitaria, il che significa che non è stazionaria.
# Ipotesi alternativa (H1): La serie temporale non ha una radice unitaria, il che significa che è stazionaria.
check_stationarity(group_df["temp"])

# split dataset into train and test

serie = group_df["temp"].values
i = int(len(serie) * 0.9)  # 54 anni per il train
j = int(len(serie) * 0.1)  # 6 anni per il test

train, test = serie[:i], serie[-j:]
assert len(train) + len(test) == len(serie)

# Statistical model SARIMA
model = pm.auto_arima(
    train,
    start_p=1,
    start_q=1,
    test="adf",
    max_p=3,
    max_q=3,
    m=12,
    start_P=0,
    seasonal=True,
    d=None,
    D=1,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)  # stepwise=False full grid
print(model.summary())
morder = model.order  # p,d,q
mseasorder = model.seasonal_order  # P,D,Q,m
fitted = model.fit(train)

# top left, errori residui che dovrebbero fluttuare attorno alla media con varianza uniforme
# top right, distribuzione normale con media 0
# bottom left, q point che dovrebbero essere allineati, la linea rossa è la gaussiana
# bottom right, acf degli errori, non troviamo stagionalità nei residui
model.plot_diagnostics(figsize=(15, 12))
plt.show()

yfore = fitted.predict(n_periods=len(test))  # forecast
ypred = fitted.predict_in_sample()
plt.plot(test, label = "Test")
plt.plot(yfore, label = "Test predictions")
plt.legend()
plt.xlabel("time")
plt.title("Sarima forecast")
plt.ylabel("temperature")
plt.show()

sarima_err = mse(yfore, test)

### NEURAL MODELS
def moving_window(data, w):
    x, y = list(), list()
    for i in range(0, len(data) - w):
        x.append(data[i : i + w])
        y.append([data[i + w]])
    return np.array(x), np.array(y)


window = 12
train_x, train_y = moving_window(train, window)
mlp_test = np.concatenate((train[-window:], test))
test_x, test_y = moving_window(mlp_test, window)


def create_mlp_model(input_dim, activation="relu", loss="mse"):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss=loss)
    return model

print("------------------\nFIT MLP MODEL\n------------------")

mlp_model = create_mlp_model(window)
mlp_model.fit(train_x, train_y, epochs=100, verbose = 0)

# evaluation and comparison between neural models
mlp_train_predict = mlp_model.predict(train_x)
mlp_test_predict = mlp_model.predict(test_x)

mlp_mse = mse(test_y, mlp_test_predict)

plt.plot(group_df["temp"] ,label="Dataset")
plt.plot(np.concatenate((np.full(window - 1, np.nan), mlp_train_predict[:, 0])), label = "Train")
plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), mlp_test_predict[:, 0])), label = "test")
plt.legend()
plt.title("MLP forecast")
plt.show()

"""
LSTM
"""


def create_LSTM_model(input_dim, activation="relu", loss="mse"):
    model = Sequential()
    model.add(LSTM(64, activation=activation, input_shape=input_dim, dropout=0.1))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss=loss)
    return model


# prepare data
n_input = 12
nfeatures = 1
std_train = train.reshape(len(train), 1)
std_trasf = StandardScaler()
std_trasf.fit(std_train)
std_train = std_trasf.transform(std_train)
generator = TimeseriesGenerator(std_train, std_train, length=n_input, batch_size=1)

std_test = std_trasf.transform(test.reshape(len(test), 1))

print("------------------\nFIT LSTM MODEL\n------------------")

# compile and train model
lstm_model = create_LSTM_model((n_input, nfeatures))
lstm_model.fit(generator, epochs=1, verbose = 0)

# model evaluation
preds_std = list()
batch = std_train[-n_input:].reshape((1, n_input, nfeatures))

for _ in range(len(test)):
    pred = lstm_model.predict(batch)[0]
    preds_std.append(pred)
    batch = np.append(batch[:, 1:, :], [[pred]], axis=1)

lstm_forecast = std_trasf.inverse_transform(preds_std)
lstm_forecast = np.transpose(lstm_forecast).squeeze()
lstm_mse = mse(lstm_forecast, test)
 


# DB MARINO
rt = dm_test(test, mlp_test_predict[:, 0], lstm_forecast, h=2, crit="MSE")
print("------------------\nDIEBOLD-MARIANO TEST\n------------------")
print(f"\nDM VALUE: {rt.DM}, p_value: {rt.p_value}\n")
print("\n" + "We accept the null hypotesys " if rt.p_value > 0.5 else "We reject the H0 hypotesis " + "\n")

# XGBoost


lookback = 12
ds_len = len(test)
dataset = pd.DataFrame()
for i in range(lookback, 0, -1):
    dataset['t-' + str(i)] = group_df["temp"].shift(i) # build dataset by columns
dataset['t'] = group_df["temp"].values
dataset = dataset[lookback:] # removes the first lookback columns
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, xtest = x[:-ds_len], x[-ds_len:]
y_train, ytest = y[:-ds_len], y[-ds_len:]

from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:squarederror',
n_estimators=1000)
model.fit(x_train, y_train)
xg_preds= model.predict(xtest)
xg_mse = mse(xg_preds, ytest)


labels = ["Test true", "LSTM predictions", "MLP predictions", "XGBoost predictions", "SARIMA predictions"]
for label, scores in zip(labels, [test, lstm_forecast, mlp_test_predict, xg_preds, yfore]):
    plt.plot(scores.flatten(), label=label)
plt.legend()
plt.title("MLP vs LSTM")
plt.show()   


print("------------------\nMODEL COMPARISON\n------------------")
print("SARIMA - MSE: %.4f" % sarima_err)
print("LSTM - MSE: %.4f" % lstm_mse)
print("MLP - MSE: %.4f" % mlp_mse)
print('XGBoost - MSE: %.4f' % xg_mse)
