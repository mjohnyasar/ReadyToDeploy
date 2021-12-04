import pandas as pd
import streamlit as st
import datetime
from PIL import Image 
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model    #  to pickle DL large files
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
#from fbprophet import Prophet

# pip install tensorflow==2.2.0 --user command
# tqm keras larin basina tensorflow. getir

st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwXqYkBpAS25PJOPnhlLweUTxnHuy5tMUHZg&usqp=CAU",width=600, caption="Could you guess the price?")

st.header(" BTC BITCOIN PRICE PREDICTON  ")
st.subheader(" What could be the BTC stock price in market tomorrow?")


datetime.datetime.now()

st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsa8W_kkvwh9q9YwU_dM_3CQDez_J8NSPytw&usqp=CAU")
st.success("For estimation Please choose one of the models below")
if st.checkbox("Facebook Prophet"):
    loaded_model = joblib.load('prophet_model.sav')  # load the model from disk
    future=loaded_model.make_future_dataframe(periods=1,freq="D")
    predicton=loaded_model.predict(future)
    guess=predicton[["ds","yhat"]][-1:]

    st.info("Estimated BTC stock price tomorrow is: ${}".format(int(guess)))

if st.checkbox("RNN model"):
    regressor=load_model('RNN_model.h5')
    #regressor = pickle.load(open('RNN_model.sav', 'rb'))
    #regressor = joblib.load('RNN_model.sav')
    scaler=joblib.load("RNN_scaler")
    df=joblib.load("RNN_df")
    df_s = scaler.fit_transform(df)     # df_s : df_scale
    X_test=np.array([df_s[-10: ]])   
    y_test_pred= regressor.predict(X_test)        #  predicted_stock_price
    guess =scaler.inverse_transform(y_test_pred)
    st.info("Estimated BTC stock price tomorrow is: ${}".format(int(guess)))




# Requirements
# streamlit==0.82.0
# pystan==2.19.1.1
# plotly==4.14.3
# fbprophet==0.7.1
# yfinance==0.1.59

