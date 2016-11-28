# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:16:46 2016

@author: Team BigInJapan
"""

import pandas as pd
import numpy as np
import datetime as dt
from functools import partial
from workalendar.europe import France


cal = France()

def extract_date(string):
    d = dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.000")
    return(d)
    
def extract_weekday(date):
    return(date.weekday())
    
def extract_hour(date):
    return(date.hour)
    
def extract_month(date):
    return(date.month)
   
def extract_year(date):
    return(date.year)

def hourlymean_past2weeks(date, y):
    
    nday_before = 7
    ysel = y.loc[(y.index > date - dt.timedelta(nday_before))]
    ysel = ysel.loc[(ysel.index < date)]
    h = extract_hour(date)
    if (len(ysel.loc[ysel.loc[:,'HOUR'] == h]) < 0):
        return(0)
    else: 
        return(ysel.loc[ysel.loc[:,'HOUR'] == h].loc[:,"CSPL_RECEIVED_CALLS"].mean())

def is_day_off(date):
#    weekday = extract_weekday(date)
    
    if (cal.is_working_day(date)):
        return 1
    else:
        return 0        
        
def lastvalue(date, y):
    nday_before = 7
    try: 
        v = y.loc[date - dt.timedelta(nday_before)]["CSPL_RECEIVED_CALLS"]
        try:
            w = y.loc[date - 2*dt.timedelta(nday_before)]["CSPL_RECEIVED_CALLS"]
        except:
            w = v
    except KeyError:
        v = y.loc[date]["CSPL_RECEIVED_CALLS"] + np.random.normal(scale = 3)
        w = v
    return((3*v + w)/4)

def load_data(path, ass, nrows = None):  

    ## Loading the data 
    col_loaded = ['DATE', 'DAY_OFF', 'WEEK_END', 
                  'ASS_ASSIGNMENT','TPER_TEAM', 'CSPL_RECEIVED_CALLS'] 
    if (nrows == None): 
        data = pd.read_csv(path, 
                       sep = ";",
                       usecols = col_loaded)
    else:
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
    
#    jour = []
    nuit = []
    
    for i in range(nrows):
        if(tper_team[i] == "Jours"): 
#            jour.append(1)
            nuit.append(0)
        else:
            nuit.append(1)
#            jour.append(0)
        
#    data['JOUR'] = jour
    data['NUIT'] = nuit
                     
    ## Selecting Data 
    
    print("Selecting used Data.")
    
    col_used = ['DATE', 'DAY_OFF', 'WEEK_END', 
                'ASS_ASSIGNMENT', 'NUIT', 'CSPL_RECEIVED_CALLS']     
    
    data = data[col_used]
    data.sort_values(["ASS_ASSIGNMENT", "DATE"], inplace = True)
               
    ## Selecting one ASS_ASSIGNEMENT (One model for each)
    
    l = len(ass)
    preproc_data = []
    rcvcall_data = []
    features_data = []
    norm = []

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
        
        hours = dates.apply(extract_hour, 1)
        preproc_data[i].loc[:,"WEEKDAY"] = dates.apply(extract_weekday, 1)
        preproc_data[i].loc[:,"HOUR"] = hours
        preproc_data[i].loc[:,"MONTH"] = dates.apply(extract_month, 1)
        preproc_data[i].loc[:,"MONTH_YEAR"] = preproc_data[i].loc[:,"MONTH"] + (dates.apply(extract_year, 1)-2011)*12 - 1
        preproc_data[i].loc[:,"DAY_OFF"] = dates.apply(is_day_off, 1)
        preproc_data[i].loc[:,"DATE"] = dates
#        preproc_data[i].index = dates
        

    #print(used_data.describe())
    
    # Paramètre moyenne et variance de la cible sur chaque jour de la semaine
    
#        m = preproc_data[i].groupby(["MONTH_YEAR", "WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.mean)
#        s = preproc_data[i].groupby(["MONTH_YEAR", "WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.std)
#        preproc_data[i].loc[:, "WEEKDAY_MEAN"] = m
#        preproc_data[i].loc[:, "WEEKDAY_STD"] = s
        
        print("Retrieving past data...")
        RCV_7DAYS = pd.DataFrame(rcvcall)
        RCV_14DAYS = pd.DataFrame(rcvcall)
        RCV_7DAYS.loc[:,"DATE"] = dates + dt.timedelta(7)
        RCV_14DAYS.loc[:,"DATE"] = dates + dt.timedelta(14)
        RCV_7DAYS = RCV_7DAYS[RCV_7DAYS.loc[:,"DATE"] < max(dates)]
        RCV_14DAYS = RCV_14DAYS[RCV_14DAYS.loc[:,"DATE"] < max(dates)]

#        y.loc[:,"HOUR"] = hours
#        y.index = dates
#        fun = partial(lastvalue, y = y)
#        preproc_data[i].loc[:, 'RCV_7DAY'] = dates.apply(fun, 1)
        preproc_data[i] = pd.merge(preproc_data[i], RCV_7DAYS, on = "DATE", suffixes = ('', '_x'), how = 'left')
        preproc_data[i] = pd.merge(preproc_data[i], RCV_14DAYS, on = "DATE", suffixes = ('', '_y'), how = 'left')        
        preproc_data[i] = pd.concat([preproc_data[i], pd.get_dummies(preproc_data[i].loc[:,'WEEKDAY'])], axis = 1, join = 'inner')
        
        preproc_data[i].rename(index=str, columns={'CSPL_RECEIVED_CALLS_x': 'RCV_7DAY','CSPL_RECEIVED_CALLS_y': 'RCV_14DAY', 0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6 : "Dimanche"}, inplace = True)
#        print(preproc_data[i])
#        print(preproc_data[i])
    #print(preproc_data) 
        preproc_data[i].index = dates
        rcvcall_data.append(preproc_data[i]['CSPL_RECEIVED_CALLS'])
        features_data.append(preproc_data[i])
        
        features_data[i].drop(["CSPL_RECEIVED_CALLS", "WEEKDAY", "DATE", "MONTH_YEAR"], 1, inplace = True)
#        features_data[i].drop(["CSPL_RECEIVED_CALLS"], 1, inplace = True)
#        print(len(preproc_data))
#        print(len(rcvcall_data))
#        print(len(features_data))
#    rcvcall_data[2].plot()
        

        ## Normalization 
        print("Normalizing the data...")
        n_call = max(rcvcall_data[i])
#        std_call = rcvcall_data[i].std()
#        n_mean = max(features_data[i].loc[:,"WEEKDAY_MEAN"])
#        n_std = max(features_data[i].loc[:,"WEEKDAY_STD"])
        norm.append(n_call)
#        norm.append(std_call)
        if n_call!=0:
#            rcvcall_data[i] /= std_call
            rcvcall_data[i] /= n_call
#            features_data[i].loc[:,'RCV_7DAY'] /= std_call
            features_data[i].loc[:,'RCV_7DAY'] /= n_call
            features_data[i].loc[:,'RCV_14DAY'] /= n_call
#        if n_mean != 0:
#            features_data[i].loc[:,'WEEKDAY_MEAN'] /= n_mean
#        if n_std != 0:
#            features_data[i].loc[:,'WEEKDAY_STD'] /= n_std
    
    
    
    return(features_data, rcvcall_data, preproc_data, norm)
    
#FOR TESTING"

if __name__ == '__main__':
    import os
    os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
    ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']
    
    features_data_test, rcvcall_data_test, preproc_data_test, norm_test = load_data("train_2011_2012_2013.csv",
                                                          ass = ass,
                                                          nrows = 200000)
    