# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:12:08 2016

@author: Team BigInJapan
"""
alpha = -0.1
import numpy as np

def linex(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    grad = (-alpha*np.exp(alpha*(y_true-y_pred)) + alpha)
    hess = alpha*alpha*np.exp(alpha*(y_true-y_pred))
    
    return grad, hess
    
def loss_linex(y_true, y_pred):

    return 'error', np.exp(alpha*(y_true-y_pred)) - alpha*(y_true-y_pred) - 1
    

    