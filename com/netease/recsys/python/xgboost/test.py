import xgboost as xgb
import pandas as pd
import time
import numpy as np
import pylab as pl

now = time.time()

dataset = pd.read_csv("data090.csv")
print dataset.head()
print dataset.describe()
print dataset["clk"].count()
print dataset.iloc[1:3, 30]
dataset["clk"].hist()
pl.show()

