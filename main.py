# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:04:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import datetime as dt
from load_data import load_data
from learn_xgb import learn_xgb

print("Importing the data...")

ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique',
           'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault',
           'Japon', 'Médical', 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 
           'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie']
l = len(ass)

features_data, rcvcall_data = load_data("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge/train_2011_2012_2013.7z/train_2011_2012_2013.csv",
                                        ass)


## Learning algorithm 
print("Fitting the data...")
models = list(range(l))
for i in range(l):
    print("Fitting n°" + str(i + 1) + "/" + str(l))
    models[i], error = learn_xgb(features_data[i], rcvcall_data[i])


## Result visualization 

n_to_vis = 3

X_test = features_data[n_to_vis][features_data[n_to_vis].index < dt.datetime(2011, 2, 16)]
y_test = rcvcall_data[n_to_vis][rcvcall_data[n_to_vis].index < dt.datetime(2011, 2, 16)]
                                                                         
y_pred = pd.DataFrame(models[n_to_vis].predict(X_test))

y_pred.index = y_test.index
y_test = pd.DataFrame(y_test)

y = pd.concat([y_test, y_pred], axis = 1)
y.columns = ["TRUE_VALUE", "PRED_VALUE"]

y.plot()

