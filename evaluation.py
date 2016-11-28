# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:55:58 2016

@author: Team BigInJapan
"""

import numpy as np
from linex import loss_linex


def evalerror(y_pred, y_test):
    
    error = loss_linex(y_test, y_pred)
    
    return(np.mean(error[1]))
    
def score_linex(estimator, X, y): 
    
    y_pred = estimator.predict(X)
    error = loss_linex(y, y_pred)    
    
    return(-np.mean(error[1]))