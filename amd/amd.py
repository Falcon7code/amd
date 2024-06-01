import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# Download historical data for AMD
amd_data = yf.download('AMD', start='2024-01-01', end=datetime.today().strftime('%Y-%m-%d'), interval='1d')
amd_data = amd_data[['Open', 'Close']]

# Ensure the index has frequency information
amd_data.index = pd.to_datetime(amd_data.index)
amd_data = amd_data.asfreq('B')

# Filter data for Year-to-Date (YTD)
current_year = datetime.today().year
amd_data_ytd = amd_data[amd_data.index.year == current_year]

# Function to fit ARIMA model and make predictions
def arima_model(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=1)  # Predict next day
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    return forecast_mean[0], forecast_conf_int

# Function to fit LSTM model and make predictions
def lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):  # Using 60 days for training
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    test_data = data[-60:].values
    test_data = test_data.reshape(-1, 1)
    test_data = scaler.transform(test_data)

    X_test = [test_data[-60:]]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price[0][0]

# Get ARIMA predictions for Open and Close prices
arima_open_forecast, arima_open_conf_int = arima_model(amd_data_ytd['Open'])
arima_close_forecast, arima_close_conf_int = arima_model(amd_data_ytd['Close'])

# Get LSTM predictions for Open and Close prices
lstm_open_predicted_price = lstm_model(amd_data_ytd['Open'])
lstm_close_predicted_price = lstm_model(amd_data_ytd['Close'])

# Plot ARIMA forecast for Open prices
plt.figure(figsize=(10, 6))
plt.plot(amd_data_ytd['Open'], label='Actual Open')
plt.plot(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), [arima_open_forecast], label='ARIMA Open Forecast', color='red', linestyle='dotted')
plt.fill_between(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), arima_open_conf_int.iloc[:, 0], arima_open_conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.title('AMD Daily Open Price Prediction using ARIMA (YTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotation for dates
plt.annotate(f'Next Day ARIMA Open: ${arima_open_forecast:.2f}', xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='red')
plt.axvline(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B')[0], color='red', linestyle='dotted')
plt.show()

# Plot LSTM predictions for Open prices
plt.figure(figsize=(10, 6))
plt.plot(amd_data_ytd['Open'].values, label='Actual Open')
plt.plot(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), [lstm_open_predicted_price], label='LSTM Open Predictions', color='green', linestyle='dotted', marker='o')
plt.title('AMD Daily Open Price Prediction using LSTM (YTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotation for dates
plt.annotate(f'Next Day LSTM Open: ${lstm_open_predicted_price:.2f}', xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='green')
plt.axvline(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B')[0], color='green', linestyle='dotted')
plt.show()

# Plot ARIMA forecast for Close prices
plt.figure(figsize=(10, 6))
plt.plot(amd_data_ytd['Close'], label='Actual Close')
plt.plot(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), [arima_close_forecast], label='ARIMA Close Forecast', color='blue', linestyle='dotted')
plt.fill_between(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), arima_close_conf_int.iloc[:, 0], arima_close_conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.title('AMD Daily Close Price Prediction using ARIMA (YTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotation for dates
plt.annotate(f'Next Day ARIMA Close: ${arima_close_forecast:.2f}', xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='blue')
plt.axvline(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B')[0], color='blue', linestyle='dotted')
plt.show()

# Plot LSTM predictions for Close prices
plt.figure(figsize=(10, 6))
plt.plot(amd_data_ytd['Close'].values, label='Actual Close')
plt.plot(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B'), [lstm_close_predicted_price], label='LSTM Close Predictions', color='purple', linestyle='dotted', marker='o')
plt.title('AMD Daily Close Price Prediction using LSTM (YTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotation for dates
plt.annotate(f'Next Day LSTM Close: ${lstm_close_predicted_price:.2f}', xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='purple')
plt.axvline(pd.date_range(start=amd_data_ytd.index[-1], periods=1, freq='B')[0], color='purple', linestyle='dotted')
plt.show()
