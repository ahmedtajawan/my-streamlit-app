1---This project is about Stock trend prediction using historical Closing stock prices.
2---Data is acquired from yahoo finance website using  yfinance python library and userinterface is created using streamlit litraray


3--- To run streamlit app cmd is to be opened inside the folder containig app.py file and executing command below command
streamlit run app.py

4--- below are the libraries used in this project
	from pandas_datareader import data as pdr
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.model_selection import train_test_split
	import tensorflow as tf
	import streamlit as st
	from keras.models import load_model
	import datetime
	import yfinance as yf 
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error
	yf.pdr_override()
5--- model is trained using google stock prices from 2010 with 100 step and that trained model is used to predict any stock using its stock ticker available on yahoo finance.

6--- folder contain 5 files 
	app.py is a streamlit app
	model.h5 is a trained model on google historical stock 
	stock trend prediction app ahmed taj is a presentation
	Stock Trend Prediction is a jupiter notebook where all the code is provided related to model training
	Readme file with all the information
