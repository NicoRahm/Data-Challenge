# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:52:19 2016

@author: Team BigInJapan
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn as skl
from linex import linex, evalerror_linex

def learn_xgb(X_train, y_train):    
    
    ## Learning algorithm 
    
    model = xgb.XGBRegressor(objective = linex, learning_rate = 0.1, silent = False, min_child_weight = 0)
    model.fit(X_train, y_train, eval_metric = evalerror_linex)
    
    return model
    