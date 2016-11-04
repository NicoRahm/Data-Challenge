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
    
    nrows = 200000
    
    data = pd.read_csv(path, 
                       sep = ";",
                       nrows = nrows)
    #data = data.set_index('DATE')
                       
    data.info()
                       
    ## Creating features jour et nuit
    
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
    
    col_used = ['DATE', 'DAY_OFF', 'WEEK_END', 'SPLIT_COD', 
                'ASS_ASSIGNMENT','JOUR', 'NUIT', 'CSPL_RECEIVED_CALLS']     
    
    preproc_data = data[col_used]
               
    
    ## Creation of Date features
    
    
    dates = preproc_data['DATE'].apply(extract_date, 1)
    
    print(dates)
    
    preproc_data.loc[:,"WEEKDAY"] = dates.apply(extract_weekday, 1)
    preproc_data.loc[:,"HOUR"] = dates.apply(extract_hour, 1)
    preproc_data.loc[:,"MONTH"] = dates.apply(extract_month, 1)
    
    #print(used_data.describe())
    
    print(preproc_data) 
    
    
    ## Selecting one ASS_ASSIGNEMENT (One model for each)
    
    used_data = preproc_data.loc[preproc_data.loc[:,"ASS_ASSIGNMENT"] == "Téléphonie", :]
    used_data.drop(["SPLIT_COD", "ASS_ASSIGNMENT"],1, inplace = True)
    print(used_data)
    
    rcvcall_data = used_data.groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
    features_data = used_data.groupby(["DATE"]).mean()
    features_data.drop("CSPL_RECEIVED_CALLS", 1, inplace = True)
    
    
    used_data = used_data.set_index('DATE')
    
    ## Cleaning the data 
    
    
    
    
    return features_data, rcvcall_data