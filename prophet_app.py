import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# prophet by Facebook
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import pickle, joblib

plt.rcParams["figure.figsize"] = (15,8)
sb.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")



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
