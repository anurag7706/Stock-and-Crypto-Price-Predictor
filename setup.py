import streamlit as st
from datetime import date
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#TODAY = "2022-04-05"
st.title('Welcome Predictors ! ')
st.sidebar.title("Give us data please ..")

# stock_pre = ('TTM', 'RELIANCE.NS', '^NSEI)')
# stock = ('Crypto' ,'Stocks',)
selected_stock = st.sidebar.selectbox('Select dataset for prediction',['Crypto','Stocks'])

if selected_stock == 'Stocks':

 stock_pre = st.sidebar.text_input(label = " Enter the abbr. of stock")
 n_years = st.sidebar.slider('Years of prediction:', min_value=1, max_value=None,)
 period = n_years * 365
 submit= st.sidebar.button("Let's Predict")

 if submit:
     with st.spinner(text='In progress'):

      @st.cache
      def load_data(ticker):
       data = yf.download(ticker, START, TODAY)
       data.reset_index(inplace=True)
       return data


      data_load_state = st.text('Loading data...')
  # data = load_data(selected_stock)
      data = load_data(stock_pre)
      data_load_state.text('Loading data... done!')

    # st.subheader('Raw data')
  # st.write(data.tail())

  # Plot raw data
  # def plot_raw_data():
  #       fig = go.Figure()
  #       fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
  #       fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
  #       fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
  #       st.plotly_chart(fig)

  # plot_raw_data()

  # Predict forecast with Prophet.
      df_train = data[['Date','Close']]
      df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

      m = Prophet()
      m.fit(df_train)
      future = m.make_future_dataframe(periods=period)
      forecast = m.predict(future)

 # Show and plot forecast
      st.subheader('Forecast data')
      st.write(forecast.tail())

      st.write(f'Forecast plot for {n_years} years')
 # fig1=m.plot(forecast)
      fig1 = plot_plotly(m, forecast)
      st.plotly_chart(fig1)

      st.write("Forecast components")
      fig2 = m.plot_components(forecast)
      st.write(fig2)

#Crypto price prediction code starts-

if selected_stock =='Crypto':
    crypto_pre = st.sidebar.text_input(label=" Enter the coin abbr.")
    st.sidebar.caption('Eg. BTC-USD,ETH-USD,etc.')
    submit= st.sidebar.button("Let's Predict")
    if submit:
     with st.spinner(text='In progress'):
      df = yf.download(tickers=crypto_pre)
      plt.plot(df.index, df['Adj Close'])
    # plt.show()

    # Train test split

      to_row = int(len(df) * 0.9)

      training_data = list(df[0:to_row]['Adj Close'])
      testing_data = list(df[to_row:]['Adj Close'])
    # split data into train and training set
    # plt.figure(figsize=(10, 6))
    # plt.grid(True)
    # plt.xlabel('Dates')
    # plt.ylabel('Closing Prices')
    # plt.plot(df[0:to_row]['Adj Close'], 'green', label='Train data')
    # plt.plot(df[to_row:]['Adj Close'], 'blue', label='Test data')
    # plt.legend()
      model_predictions = []
      n_test_obser = len(testing_data)
     # st.write(n_test_obser)

     # yhat = list(output[0])[0]

      for i in range(n_test_obser):
       model = ARIMA(training_data, order=(4, 1, 0))
       model_fit = model.fit()
       output = model_fit.forecast()
       yhat = output
       model_predictions.append(yhat)
       actual_test_value = testing_data[i]
       training_data.append(actual_test_value)
# st.write(yhat)
#     st.write(model_fit.summary())
#      plt.figure(figsize=(15, 9))
#      plt.grid(True)
      fig=plt.figure(figsize=(15, 9))
      plt.grid(True)
      date_range = df[to_row:].index
      plt.title('Price Prediction')
      plt.xlabel('Date')
      plt.ylabel('Price')
      plt.plot(date_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
      plt.plot(date_range,testing_data, color='red', label='Actual Price')
      plt.legend()
      plt.show()
      st.pyplot(fig)
      st.success('Done')

     # fig=go.Figure()


