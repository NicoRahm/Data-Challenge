# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:52:19 2016

@author: Team BigInJapan
"""
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from linex import linex, loss_linex
from evaluation import score_linex

def learn_xgb(X_train, y_train):    
    
    ## Learning algorithm 
    
    model = xgb.XGBRegressor(max_depth = 10, 
                             n_estimators = 50, 
                             objective =  linex, 
                             learning_rate = 0.1, 
                             silent = False, 
                             min_child_weight = 0.0000001)
    errors = 0
    errors = cross_val_score(model,X_train, y_train, cv = 5, scoring = score_linex)  
    
    print("Cross-Validation errors : " + str(errors))
    
    model.fit(X_train, y_train, eval_metric = loss_linex)
    
    return model, errors
    