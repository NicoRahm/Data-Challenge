# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:12:08 2016

@author: Team BigInJapan
"""

import numpy as np

def linex(y_true, y_pred):
    l = len(y_pred)
    alpha = -0.1
    
    grad = (-alpha*np.exp(alpha*(y_true-y_pred)) + alpha)
    hess = alpha*alpha*np.exp(alpha*(y_true-y_pred))
    
    return grad, hess
    
def evalerror_linex(y_true, y_pred):
    alpha = -0.1
    return 'error', np.exp(alpha*(y_true-y_pred)) - alpha*(y_true-y_pred) - 1
    