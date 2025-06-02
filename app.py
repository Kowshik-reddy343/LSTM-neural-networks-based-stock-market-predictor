import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
# Load the pre-trained model
model = load_model('stock prediction with keras.keras')
st.header('Stock Price Prediction')
stock=st.text_input('Enter Stock Ticker', 'GOOG')
start='2012-1-1'
end='2019-12-31'
# Fetch the stock data
data = yf.download(stock, start, end)
st.subheader(f"Stock Data for {stock}")
st.write(data)
if data.empty:
    st.error("No data found for the given stock ticker.")
else:
    dtata_train= pd.DataFrame(data.Close[0:int(len(data)*0.8)])
    dtata_test = pd.DataFrame(data.Close[int(len(data)*0.8):len(data)])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    pas_100_days= dtata_train.tail(100)
    dtata_test = pd.concat([pas_100_days, dtata_test],ignore_index=True)
    data_test_scale = scaler.fit_transform(dtata_test)
    st.subheader("MA50")
    ma_50_days= data['Close'].rolling(window=50).mean()
    fig1= plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(ma_50_days, label='MA50', color='orange')
    plt.title(f"{stock} Stock Price and MA50")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig1)
    st.subheader("price vs MA50 vs MA100")
    ma_100_days = data['Close'].rolling(window=100).mean()
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(ma_50_days, label='MA50', color='orange')
    plt.plot(ma_100_days, label='MA100', color='green')
    plt.title(f"{stock} Stock Price, MA50, and MA100")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig2)
    st.subheader("price vs MA50 vs MA100 vs MA200")
    ma_200_days = data['Close'].rolling(window=200).mean()
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(ma_50_days, label='MA50', color='orange')
    plt.plot(ma_100_days, label='MA100', color='green')
    plt.plot(ma_200_days, label='MA200', color='red')
    plt.title(f"{stock} Stock Price, MA50, MA100, and MA200")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig3)
    x=[]
    y=[]
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)
    if x.shape[0] > 0:
        predict = model.predict(x)
        scale=1/scaler.scale_
        predict = predict * scale[0]
        y=y*scale
        st.subheader(f"Predicted Stock Prices for {stock}")
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(y, label='Actual Price')
        plt.plot(predict, label='Predicted Price', color='red')
        plt.title(f"{stock} Stock Price Prediction")
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.show()
        st.pyplot(fig4)
    else:
        st.error("Not enough data to make predictions.")