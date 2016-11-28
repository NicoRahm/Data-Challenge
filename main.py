# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:04:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import datetime as dt
from load_data import load_data
from learn_xgb import learn_xgb
import os
import saveload_models as slmod

# Working directory to change 
os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
name = input("Enter the name of the models to save (Enter to skip saving):")

import_data = input("Do I have to import the data?y/[n] ")

ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']
l = len(ass)

if (import_data == 'y'):

    print("Importing the data...")
    features_data, rcvcall_data, preproc_data, norm = load_data("train_2011_2012_2013.csv",
                                                          ass)

    for i in range(len(ass)):
        features_data[i].drop(["CSPL_RECEIVED_CALLS"], 1, inplace = True)
## Learning algorithm 
print("Fitting the data...")
models = list(range(l))
t = range(len(ass))
for i in t:
    print("Fitting n°" + str(i + 1) + "/" + str(l) + " : " + ass[i])
    models[i], error = learn_xgb(features_data[i], rcvcall_data[i])
#    for j in range(24):
#        models[i][j], error = learn_xgb(features_data[i].loc[features_data[i].loc[:,"HOUR"] == j], rcvcall_data[i].loc[features_data[i].loc[:,"HOUR"] == j])


models = pd.DataFrame(models)
models.index = ass
models = models.rename(index = str, columns = {0:"Models"})

if (name != ''):
    print("Saving models...")
    slmod.save_multiple_xgb(models, name)
    
## Result visualization 

print("Visualizing results...")
vis = [17]
for i in vis:
    print("Results for " + ass[i])
    n_to_vis = i
    ass_to_vis = ass[n_to_vis]
    
    X_test = features_data[n_to_vis][features_data[n_to_vis].index > dt.datetime(2013, 10, 18)]
    y_test = rcvcall_data[n_to_vis][rcvcall_data[n_to_vis].index > dt.datetime(2013, 10, 18)]
    X_test = X_test[X_test.index < dt.datetime(2013, 11, 3)]     
    y_test = y_test[y_test.index < dt.datetime(2013, 11, 3)]                                                                       
    y_pred = pd.DataFrame(models.loc[ass_to_vis]['Models'].predict(X_test))
    
    y_pred.index = y_test.index
    y_test = pd.DataFrame(y_test)
    
    y = pd.concat([y_test, y_pred], axis = 1)
    y.columns = ["TRUE_VALUE", "PRED_VALUE"]
    
    y.loc[:,"PRED_VALUE"]*=norm[n_to_vis]
    y.loc[:,"TRUE_VALUE"]*=norm[n_to_vis]
    
    y.plot()




