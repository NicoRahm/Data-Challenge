#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:49:57 2016

@author: Team BigInJapan
"""

import saveload_models as slmod
import learn_xgb
import load_test as lt
import xgboost as xgb
import sklearn as skl
import os
import pandas as pd
import write_test_answer as wta

ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']

name = input("Name of the models to load (Enter to skip) : ")

if (name != ''):
    print("Loading models")
    models = slmod.load_multiple_xgb(ass, name)
    
    
  
load = input("Load test data?(y/[n])")

if (load == 'y'):
    # Working directory to change 
    os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
    
    print("Loading test data...")

    test = lt.read_file_content(None, "train_2011_2012_2013.csv", "submission.txt", rcvcall_data, preproc_data)





print("Predicting...")
prediction = []

l = len(ass)
for i in range(l): 
    print("Prediction for model n°" + str(i+1) + "/" + str(l) + " : " + ass[i])
#    preds = []
    prediction.append(models.loc[ass[i]][0].predict(test.loc[ass[i]][0]))
#    for j in range(24):
#        t = test.loc[ass[i]][0]
#        th = t[t.loc[:,"HOUR"] == j]
#        pred_j = models.loc[ass[i]][j].predict(th)
#        pred_j = pd.DataFrame(pred_j, index=th.index)
#        preds.append(pred_j)
        
#    prediction.append(pd.concat(preds))
#    print(prediction[i])
    prediction[i] = pd.DataFrame(prediction[i], index = test.loc[ass[i]][0].index)
    prediction[i] = prediction[i]*norm[i]*1.7
    prediction[i] = prediction[i].apply(float, 1)

prediction = pd.DataFrame(prediction, index = ass)
print("Done\n")
print("Writing the submission file")
os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")

wta.updateSub(prediction, "submission.txt")


