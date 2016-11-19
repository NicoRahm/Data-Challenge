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
    

def load_data(path, ass):  

    ## Loading the data 
    col_loaded = ['DATE', 'DAY_OFF', 'WEEK_END', 
                  'ASS_ASSIGNMENT','TPER_TEAM', 'CSPL_RECEIVED_CALLS'] 
    nrows = 200000
    
    data = pd.read_csv(path, 
                       sep = ";",
                       usecols = col_loaded,
                       nrows = nrows) 
                       
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
    
    data = data[col_used]
    data.sort_values(["ASS_ASSIGNMENT", "DATE"], inplace = True)
               
    ## Selecting one ASS_ASSIGNEMENT (One model for each)
    
    l = len(ass)
    preproc_data = []
    rcvcall_data = []
    features_data = []

    for i in range(l):
        
        print("Preprocessing n°" + str(i + 1) + "/" + str(l))
    
        assignement = ass[i]
        print("Selecting one ASS_ASSIGNEMENT : " + assignement)
            
        preproc_data.append(data.loc[data.loc[:,"ASS_ASSIGNMENT"] == assignement, :])
        preproc_data[i].drop(["ASS_ASSIGNMENT"],1, inplace = True)
        
        nrows = len(preproc_data[i].index)  
        print(str(nrows) + " rows to process.")
        
        print("Grouping by dates...")
        rcvcall = preproc_data[i].groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
        preproc_data[i] = preproc_data[i].groupby(["DATE"]).mean()
        preproc_data[i].loc[:, 'CSPL_RECEIVED_CALLS'] = rcvcall   
        preproc_data[i].loc[:, 'DATE'] = preproc_data[i].index
        
        nrows = len(preproc_data[i])
        print(str(nrows) + " rows to process.")
#    print(preproc_data)
    
    ## Feature engineering
    
        print("Feature engineering...")    
    
        # Paramètres pour les dates
        dates = preproc_data[i]['DATE'].apply(extract_date, 1)
        
        #print(dates)
        
        preproc_data[i].loc[:,"WEEKDAY"] = dates.apply(extract_weekday, 1)
        preproc_data[i].loc[:,"HOUR"] = dates.apply(extract_hour, 1)
        preproc_data[i].loc[:,"MONTH"] = dates.apply(extract_month, 1)
        
        preproc_data[i].loc[:,"DATE"] = dates
    
    #print(used_data.describe())
    
    # Paramètre moyenne et variance de la cible sur chaque jour de la semaine
    
        m = preproc_data[i].groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.mean)
        s = preproc_data[i].groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.std)
        preproc_data[i].loc[:, "WEEKDAY_MEAN"] = m
        preproc_data[i].loc[:, "WEEKDAY_STD"] = s

    #print(preproc_data) 
        rcvcall_data.append(preproc_data[i]['CSPL_RECEIVED_CALLS'])
        features_data.append(preproc_data[i])
        features_data[i].drop(["CSPL_RECEIVED_CALLS"], 1, inplace = True)
        
#        print(len(preproc_data))
#        print(len(rcvcall_data))
#        print(len(features_data))
#    rcvcall_data[2].plot()
    ## Cleaning the data 
    
    
    
    
    return features_data, rcvcall_data, preproc_data
    
#FOR TESTING"

if __name__ == '__main__':
    import os
    os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
    ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']
    
    features_data, rcvcall_data, preproc_data = load_data("train_2011_2012_2013.csv",
                                                      ass)
    