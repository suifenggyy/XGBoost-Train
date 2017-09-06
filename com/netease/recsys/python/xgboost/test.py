import xgboost as xgb
import pandas as pd
import time
import numpy as np
import pylab as pl

now = time.time()

dataset = pd.read_csv("train.csv")
print dataset.head()
print dataset.describe()
print dataset['33'].count()

dataset['33'].hist()
pl.show()