# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:16:46 2016

@author: Team BigInJapan
WOOP WOOP
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
import os
from workalendar.europe import France
import string


cal = France()

def get_file_content(filename):

	with open(filename, "r") as fichier_test:
		content = fichier_test.readlines()
		print("Reading File")
		return content

def read_file_content(nrows, path_train, filename, rcvcall_data, preproc_data):

	ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']

	print("Retrieving mean and std arrays")
#	mean_ass, std_ass, rcvcalls, preproc = get_day_mean_std(nrows, path_train, ass)
	rcvcalls = rcvcall_data  
	preproc = preproc_data   
	test_matrix = []
	found_ass = []

	content = get_file_content(filename)

	skip = True

	print("Starting the features extraction")

	#Initializing the cols
	
	date_list = []
	notFound = 0



	for line in content:

		if (skip):
			print("Skipping first line")
			skip = False
		else:

			feature = []
			splitted = line.split("	")

			date = extract_date(splitted[0])
			date.replace(microsecond = 000)
			weekday = extract_weekday(date)
			hour = extract_hour(date)
			month = extract_month(date)
			year = extract_year(date)

			day_off = is_day_off(splitted[0], weekday)

			if (weekday == 5 or weekday == 6):
				we = 1
			else:
				we = 0

			jour, nuit = return_day_night(splitted[0])

			assignment = splitted[1]

			if (assignment not in found_ass):
				found_ass.append(assignment)

			try:
				ass_index = ass.index(assignment)
#				mean = mean_ass[ass_index][month+year]
#				std = std_ass[ass_index][month+year]
#				sel = preproc[ass_index].loc[preproc_test[ass_index].loc[:,"MONTH_YEAR"] == month+13*(year-2011)][preproc_test[ass_index].loc[:,"WEEKDAY"] == weekday]
#				mean = mean_ass[ass_index][month+12*(year-2011)-1, weekday]
#				std = std_ass[ass_index][month+12*(year-2011)-1, weekday]
				week_before = date - dt.timedelta(7)
				two_weeks_before = date - dt.timedelta(14)

				#print(week_before)
#				try:
#					#test = dt.datetime(2011,1,1,0,30,0)
#					alpha = rcvcalls[ass_index].loc[week_before]['CSPL_RECEIVED_CALLS']
#					try:
#						beta = rcvcalls[ass_index].loc[two_weeks_before]['CSPL_RECEIVED_CALLS']
#					except:
#						beta = 0
#					#print(alpha)
#
#				except KeyError:
#					alpha = 0
#					beta = 0
#					notFound += 1
				#print(alpha)
#				alpha = (3*alpha + beta)/4
				feature.append(date)
				feature.append(day_off)
				feature.append(we)
				feature.append(assignment)
#				feature.append(jour)
				feature.append(nuit)
				feature.append(weekday)
				feature.append(hour)
				feature.append(month)
#				feature.append(mean)
#				feature.append(std)
#				feature.append(alpha)
				date_list.append(date)
				
				test_matrix.append(feature)

			except ValueError:
				print("I found smth that is not in the assignment list : %s" % assignment)

    
    
	print("Done\n")
	#print (found_ass)
	print("There are %d vectors" % len(test_matrix))
	print("We didn't find %d dates" % notFound)

	dataFrame = pd.DataFrame(test_matrix, index = date_list,  columns = ['DATE', 'DAY_OFF', 'WEEK_END', 
                'ASS_ASSIGNMENT', 'NUIT', 'WEEKDAY', 'HOUR', 'MONTH'])

	test_matrix_by_ass = []
	i = 0
	for ass_assign in ass:
		test_matrix_by_ass.append(dataFrame.loc[dataFrame.loc[:,"ASS_ASSIGNMENT"] == ass_assign, :])
		RCV_7DAYS = pd.DataFrame(rcvcalls[i])
		RCV_14DAYS = pd.DataFrame(rcvcalls[i])
		RCV_7DAYS.loc[:,"DATE"] = rcvcalls[i].index + dt.timedelta(7)
		RCV_14DAYS.loc[:,"DATE"] = rcvcalls[i].index + dt.timedelta(14)
		RCV_7DAYS = RCV_7DAYS[RCV_7DAYS.loc[:,"DATE"] < max(rcvcalls[i].index)]
		RCV_14DAYS = RCV_14DAYS[RCV_14DAYS.loc[:,"DATE"] < max(rcvcalls[i].index)]       
		
		test_matrix_by_ass[i] = pd.merge(test_matrix_by_ass[i], RCV_7DAYS, on = "DATE", suffixes = ('', '_x'), how = 'left')
		test_matrix_by_ass[i] = pd.merge(test_matrix_by_ass[i], RCV_14DAYS, on = "DATE", suffixes = ('', '_y'), how = 'left')        
		test_matrix_by_ass[i] = pd.concat([test_matrix_by_ass[i], pd.get_dummies(test_matrix_by_ass[i].loc[:,'WEEKDAY'])], axis = 1, join = 'inner')
		for j in range(7):
			try:
				test_matrix_by_ass[i].loc[:,j]
			except KeyError:
				test_matrix_by_ass[i].loc[:,j] = pd.DataFrame([0 for i in range(len(rcvcalls[i].index))])		        
		test_matrix_by_ass[i].rename(index=str, columns={'CSPL_RECEIVED_CALLS': 'RCV_7DAY', 'CSPL_RECEIVED_CALLS_y': 'RCV_14DAY', 0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6 : "Dimanche"}, inplace = True)
		test_matrix_by_ass[i].index = test_matrix_by_ass[i].loc[:,'DATE']
		test_matrix_by_ass[i].drop(['DATE', 'WEEKDAY', 'ASS_ASSIGNMENT'], 1, inplace = True)	 
		test_matrix_by_ass[i] =  test_matrix_by_ass[i].loc[:,['DAY_OFF', 'WEEK_END', 'NUIT', 'HOUR', 'MONTH', 'RCV_7DAY', 'RCV_14DAY', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']]
		i+=1
	a = pd.DataFrame(test_matrix_by_ass, index = ass)
	#print(a.loc[ass[4], :])
	return a
			


def return_day_night(time_full):

	time = time_full.split(" ")
	hour_min = (time[0].split("-"))
	hour_proper = int(hour_min[0])
	minutes = int(hour_min[1])

	if (hour_proper >= 7 and minutes >= 30 and hour_proper < 23):
		jour = 1
		nuit = 0
	if (hour_proper >= 23 and minutes <= 30):
		jour = 1
		nuit = 0
	if (hour_proper >= 23 and minutes >= 30):
		jour = 0
		nuit = 1
	if (hour_proper <= 6):
		jour = 0
		nuit = 1
	if (hour_proper == 7  and minutes < 30):
		jour = 0
		nuit = 1

	return jour, nuit


def is_day_off(date, weekday):

	
	splitted = ((date.split(" "))[0]).split("-")
	#print(splitted)
	y = int(splitted[0])
	m = int(splitted[1])
	d = int(splitted[2])

	if (cal.is_working_day(dt.date(y,m,d))):
		return 1
	else:
		return 0







# Everything below allows to retrieve the array Asssignment - Weekday_Mean and Assignment - Weekday_STD 	

def extract_date(string):
    d = dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")
    return(d)
    
def extract_weekday(date):
    return(date.weekday())
    
def extract_hour(date):
    return(date.hour)
    
def extract_month(date):
    return(date.month)

def extract_year(date):
    return(date.year)

def get_day_mean_std(nrows, path, assignments):

	col_loaded = ['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']

	data = pd.read_csv(path, sep = ";",usecols = col_loaded, nrows = nrows)
	nrows = len(data.index)  #In case smth went funny 

	data.sort_values(["ASS_ASSIGNMENT", "DATE"], inplace = True)

	#print(data)

	n_ass = len(assignments)
	preproc_data = []
	mean_ass = []
	std_ass = []
	rcvcalls = []

	for i in range(n_ass):

		assignment = assignments[i]
		preproc_data.append(data.loc[data.loc[:,"ASS_ASSIGNMENT"] == assignment, :])
		preproc_data[i] = preproc_data[i].drop(["ASS_ASSIGNMENT"],1)

		nrows = len(preproc_data[i].index)
		#print(str(nrows) + " rows to process.")

		#print("Grouping by dates...")
		rcvcall = preproc_data[i].groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
		#rcvcall.set_index(['DATE'])
		preproc_data[i] = preproc_data[i].groupby(["DATE"]).mean()
		preproc_data[i].loc[:, 'CSPL_RECEIVED_CALLS'] = rcvcall
		preproc_data[i].loc[:, 'DATE'] = preproc_data[i].index
		nrows = len(preproc_data[i])
		#print("Feature engineering...")

		dates = preproc_data[i]['DATE'].apply(extract_date, 1)
		preproc_data[i].loc[:,"WEEKDAY"] = dates.apply(extract_weekday, 1)
		preproc_data[i].loc[:,"HOUR"] = dates.apply(extract_hour, 1)
		preproc_data[i].loc[:,"MONTH"] = dates.apply(extract_month, 1)

		preproc_data[i].loc[:,"DATE"] = dates
		preproc_data[i].loc[:,"MONTH_YEAR"] = preproc_data[i].loc[:,"MONTH"] + (dates.apply(extract_year, 1) - 2011)*12 - 1
		m = preproc_data[i].groupby(["MONTH_YEAR", "WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.mean)
		s = preproc_data[i].groupby(["MONTH_YEAR", "WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.std)
		preproc_data[i].loc[:, "WEEKDAY_MEAN"] = m
		preproc_data[i].loc[:, "WEEKDAY_STD"] = s
		#print preproc_data[i]
		rcvcall = pd.DataFrame(rcvcall, index = dates)

#		mean, std = extract_mean_std(preproc_data[i])
		mean = np.zeros((36,7)) 
		std = np.zeros((36,7)) 
		for j in range(36):
  			for k in range(7):
  				sel = preproc_data[i].loc[preproc_data[i].loc[:,"MONTH_YEAR"] == j][preproc_data[i].loc[:,"WEEKDAY"] == k] 
#  				print(sel)      
  				mean[j,k] = sel["WEEKDAY_MEAN"].mean()
  				std[j,k] = sel["WEEKDAY_STD"].mean()
		max_mean = mean.max()
		max_std = std.max()
		max_rcvcall = max(rcvcall['CSPL_RECEIVED_CALLS'])

		if max_mean != 0:
			mean_ass.append(mean/max_mean)
		else:
			mean_ass.append(mean)

		if (max_std != 0):
			std_ass.append(std/max_std)
		else:
			std_ass.append(std)

		if (max_rcvcall != 0):		
			rcvcalls.append(rcvcall/max_rcvcall)
		else:
			rcvcalls.append(rcvcall)

		#print(rcvcall)

	#print (mean_ass)
	#print(std_ass)

	#rcvcall[3.sort_values(["DATE"], inplace = True)
	#print (rcvcall)

	return mean_ass, std_ass, rcvcalls, preproc_data


def extract_mean_std(data):

	weekday = 0
	mean = []
	std = []



	for i in range(len(data.index)):
		if (weekday == 7):
			break
		if (data['WEEKDAY'].values[i] == weekday):
			#print("FOUND")
			mean.append(data['WEEKDAY_MEAN'].values[i])
			std.append(data['WEEKDAY_STD'].values[i])
			weekday += 1

	while (weekday != 7):
		mean.append(0)
		std.append(0)
		weekday += 1

	return mean, std


#FOR TESTING"

if __name__ == '__main__':
	ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']
	os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
#	m_test, std_test, rcvcalls_test, preproc_test = get_day_mean_std(20000, "train_2011_2012_2013.csv", ass) 
	read_file_content(20000, "train_2011_2012_2013.csv", "submission.txt", rcvcall_data, preproc_data)