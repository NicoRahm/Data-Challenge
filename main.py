# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:04:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import numpy as np
import datetime as dt
import sklearn as skl

from load_data import load_data

from learn_xgb import learn_xgb
from evaluation import evalerror

print("Importing the data...")
features_data, rcvcall_data = load_data("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge/train_2011_2012_2013.7z/train_2011_2012_2013.csv")


## Learning algorithm 
print("Fitting the data...")
model, error = learn_xgb(features_data, rcvcall_data)


## Result visualization 

X_test = features_data[features_data.index < dt.datetime(2011, 2, 16)]
y_test = rcvcall_data[rcvcall_data.index < dt.datetime(2011, 2, 16)]
                                                                         
y_pred = pd.DataFrame(model.predict(X_test))

y_pred.index = y_test.index
y_test = pd.DataFrame(y_test)

y = pd.concat([y_test, y_pred], axis = 1)
y.columns = ["TRUE_VALUE", "PRED_VALUE"]

y.plot()




