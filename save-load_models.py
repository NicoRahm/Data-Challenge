#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 00:43:44 2016

@author: Team BigInJapan
"""


from sklearn.externals import joblib

default_dir = "/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge"

# Save/Load for one model
def save_xgb(model, 
             name, 
             dir_path = default_dir):
    
    joblib.dump(model, dir_path + "/" + name + ".pkl")

def load_xgb(name, 
             dir_path = default_dir):
    
    return(joblib.load(dir_path + "/" + name + ".pkl"))

# Save/Load for a list of models
def save_multiple_xgb(models, 
                      name, 
                      dir_path = default_dir):
    
    l = len(models)
    for i in range(l): 
        save_xgb(models[i], name + "-" + str(i), dir_path)

def load_multiple_xgb(n_models, 
                      name, 
                      dir_path = default_dir):
    
    models = []
    for i in range(n_models):
        models.append(load_xgb(name + "-" + str(i), dir_path))
    
    return(models)

if __name__ == '__main__':

    import xgboost as xgb
    model = xgb.XGBRegressor(max_depth = 10, 
                             n_estimators = 500, 
                             learning_rate = 0.1, 
                             silent = False, 
                             min_child_weight = 0.1)
    
    
    save_xgb(model, "Test")
    
    model_loaded = load_xgb("Test")
    
    models = [model, model]

    save_multiple_xgb(models, "Test_mul")
    models_loaded = load_multiple_xgb(2, "Test_mul")