import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

df=pd.read_csv('crime-statistics-pro3.csv',index_col='DATE',parse_dates=True)
df=df.dropna()

crime_index = 0
crime_name = ["MURDER", "HOMICIDE", "PHYSICAL INJURY", "RAPE", "ROBBERY", "THEFT", "CARNAPPING MV", "CARNAPPING MC"]
crime = crime_name[crime_index]

df[crime].plot(figsize=(12,5), label=crime.capitalize(), legend=True, title=crime.capitalize() + ' Cases')