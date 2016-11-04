# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:55:58 2016

@author: Team BigInJapan
"""

import pandas as pd
import numpy as np
import sklearn as skl
from linex import linex, evalerror_linex


def evalerror(y_pred, y_test):
    
    error = evalerror_linex(y_test, y_pred)
    
    return(np.mean(error[1]))
    