#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:09:16 2016

@author: Team BigInJapan
"""

import statsmodels.api as sm
from linex import linex, loss_linex
from evaluation import score_linex

def learn_ARIMA(y_train, freq = None):    
    
    ## Learning algorithm 
    model = sm.tsa.ARIMA(y_train, (2,0,0), freq = freq).fit()
    
    return model

    
# For Testing

if (__name__ == '__main__'):
    import pandas as pd
    import numpy as np
    import math
    import datetime as dt
    
    n = 20000
    
    t = np.array(range(n))*2*math.pi/50
    y_train = pd.DataFrame(t)
    y_train = y_train.apply(math.cos,1)
    samples = np.random.normal(0, 0.02, size=n)
    y_train += samples
    d = dt.datetime(2011,1,1,0,0)
    dates = []
    for i in range(n):
        dates.append(d)
        d += dt.timedelta(1)
        
    y_train.index = dates
    model_test_ARIMA = learn_ARIMA(y_train)
    
    pred = model_test_ARIMA.predict(1,100,True)
    pred.plot()
    y_train[1:100].plot()
    