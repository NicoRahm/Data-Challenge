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


#features_data, rcvcall_data = load_data("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge/train_2011_2012_2013.7z/train_2011_2012_2013.csv")


## Splitting for crossvalidation 
    
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = skl.cross_validation.train_test_split(features_data, rcvcall_data, test_size=test_size, random_state=seed)

## Learning algorithm 

model = learn_xgb(X_train, y_train)

y_pred = model.predict(X_test)

## Model evaluation

print("Error : " + str(evalerror(y_pred, y_test)))





