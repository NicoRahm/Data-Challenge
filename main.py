# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:04:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import numpy as np


## Loading the data 

nrows = 200000

data = pd.read_csv("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge/train_2011_2012_2013.7z/train_2011_2012_2013.csv", 
                   sep = ";",
                   nrows = nrows)
#data = data.set_index('DATE')
                   
## Création des feature jour et nuit

tper_team = data['TPER_TEAM'].values.tolist()

jour = []
nuit = []

for i in range(nrows):
    if(tper_team[i] == "Jours"): 
        jour.append(1)
        nuit.append(0)
    else:
        nuit.append(1)
        jour.append(0)
    
data['JOUR'] = jour
data['NUIT'] = nuit
                 
## Selecting Data 

col_used = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'SPLIT_COD', 
            'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS', 'JOUR', 'NUIT']     

used_data = data[col_used]
print(used_data)            





print(used_data.describe())



## Cleaning the data 


