# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:16:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import numpy as np
import datetime as dt


def extract_date(string):
    d = dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.000")
    return(d)
    
def extract_weekday(date):
    return(date.weekday())
    
def extract_hour(date):
    return(date.hour)
    
def extract_month(date):
    return(date.month)
    

def load_data(path):  

    ## Loading the data 
    col_loaded = ['DATE', 'DAY_OFF', 'WEEK_END', 
                  'ASS_ASSIGNMENT','TPER_TEAM', 'CSPL_RECEIVED_CALLS'] 
    nrows = 200000
    
    data = pd.read_csv(path, 
                       sep = ";",
                       usecols = col_loaded) 
                       
    nrows = len(data.index)
    #data = data.set_index('DATE')
                       
    print(str(nrows) + " rows imported.")
                       
    #data.info()
                       
    ## Creating features jour et nuit
                       
    print("Creating features Jour et Nuit.")
    
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
    
    print("Selecting used Data.")
    
    col_used = ['DATE', 'DAY_OFF', 'WEEK_END', 
                'ASS_ASSIGNMENT','JOUR', 'NUIT', 'CSPL_RECEIVED_CALLS']     
    
    preproc_data = data[col_used]
    preproc_data.sort_values(["ASS_ASSIGNMENT", "DATE"], inplace = True)
               
        ## Selecting one ASS_ASSIGNEMENT (One model for each)
    
    preproc_data = preproc_data.loc[preproc_data.loc[:,"ASS_ASSIGNMENT"] == "Téléphonie", :]
    preproc_data.drop(["ASS_ASSIGNMENT"],1, inplace = True)
    
    nrows = len(preproc_data.index)   
    print(str(nrows) + " rows to process.")
    
    rcvcall_data = preproc_data.groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
    preproc_data = preproc_data.groupby(["DATE"]).mean()
    preproc_data.loc[:, 'CSPL_RECEIVED_CALLS'] = rcvcall_data   
    preproc_data.loc[:, 'DATE'] = preproc_data.index
    
    nrows = len(preproc_data)
    print(str(nrows) + " rows to process.")
    print(preproc_data)
    
    ## Feature engineering
    
    print("Feature engineering...")    
    
    # Paramètres pour les dates
    dates = preproc_data['DATE'].apply(extract_date, 1)
    
    #print(dates)
    
    preproc_data.loc[:,"WEEKDAY"] = dates.apply(extract_weekday, 1)
    preproc_data.loc[:,"HOUR"] = dates.apply(extract_hour, 1)
    preproc_data.loc[:,"MONTH"] = dates.apply(extract_month, 1)
    
    preproc_data.loc[:,"DATE"] = dates
    
    #print(used_data.describe())
    
    # Paramètre moyenne et variance de la cible sur chaque jour de la semaine
    
    m = preproc_data.groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.mean)
    s = preproc_data.groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.std)
    preproc_data.loc[:, "WEEKDAY_MEAN"] = m
    preproc_data.loc[:, "WEEKDAY_STD"] = s
    #print(preproc_data) 
    

    #print(used_data)
    
    
    rcvcall_data = preproc_data.groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
    features_data = preproc_data.groupby(["DATE"]).mean()
    features_data.drop(["CSPL_RECEIVED_CALLS"], 1, inplace = True)
    
    
    rcvcall_data.plot()
    ## Cleaning the data 
    
    
    
    
    return features_data, rcvcall_data