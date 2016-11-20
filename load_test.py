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


def get_file_content(filename):

	with open(filename, "r") as fichier_test:
		content = fichier_test.readlines()
		print("Reading File")
		return content

def read_file_content(path_train, filename, nrows = None):

	ass = ['CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 
	'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical',
	 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 
	 'Tech. Total', 'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']

	print("Retrieving mean and std arrays")
	mean_ass, std_ass = get_day_mean_std(nrows, path_train, ass)

	test_matrix = []
	found_ass = []

	content = get_file_content(filename)

	skip = True

	print("Starting the features extraction")

	#Initializing the cols
	
	date_list = []



	for line in content:

		if (skip):
			print("Skipping first line")
			skip = False
		else:

			feature = []
			splitted = line.split("	")

			date = extract_date(splitted[0])
			weekday = extract_weekday(date)
			hour = extract_hour(date)
			month = extract_month(date)

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
				mean = mean_ass[ass_index][weekday]
				std = std_ass[ass_index][weekday]

				feature.append(date)
				feature.append(day_off)
				feature.append(we)
				feature.append(assignment)
				feature.append(jour)
				feature.append(nuit)
				feature.append(weekday)
				feature.append(hour)
				feature.append(month)
				feature.append(mean)
				feature.append(std)

				date_list.append(date)

				test_matrix.append(feature)

			except ValueError:
				print("I found smth that is not in the assignment list : %s" % assignment)
	
	print("Done\n")
	#print (found_ass)
	print("There are %d vectors" % len(test_matrix))

	dataFrame = pd.DataFrame(test_matrix, index = date_list,  columns = ['DATE', 'DAY_OFF', 'WEEK_END', 
                'ASS_ASSIGNMENT','JOUR', 'NUIT', 'WEEKDAY', 'HOUR', 'MONTH', 'WEEKDAY_MEAN', 'WEEKDAY_STD'])
	
	test_matrix_by_ass = []
	i = 0
	for ass_assign in ass:
#		print(ass_assign)
		test_matrix_by_ass.append(dataFrame.loc[dataFrame.loc[:,"ASS_ASSIGNMENT"] == ass_assign, :])
		test_matrix_by_ass[i].drop(['DATE', 'ASS_ASSIGNMENT'], 1, inplace = True)
		i+=1

	test_matrix_by_ass = pd.DataFrame(test_matrix_by_ass)
	test_matrix_by_ass.index = ass         
#	print(test_matrix_by_ass)

	return test_matrix_by_ass
			


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

	cal = France()
	splitted = ((date.split(" "))[0]).split("-")
	#print(splitted)
	y = int(splitted[0])
	m = int(splitted[1])
	d = int(splitted[2])

	if (cal.is_working_day(dt.date(y,m,d)) and (weekday != 6)):
		return 0
	else:
		return 1







# Everything below allows to retrieve the array Asssignment - Weekday_Mean and Assignment - Weekday_STD 	

def extract_date(string):
    d = dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.000")
    return(d)
    
def extract_weekday(date):
    return(date.weekday())
    
def extract_hour(date):
    return(date.hour)
    
def extract_month(date):
    return(date.month)

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

	for i in range(n_ass):

		assignment = assignments[i]
		preproc_data.append(data.loc[data.loc[:,"ASS_ASSIGNMENT"] == assignment, :])
		preproc_data[i] = preproc_data[i].drop(["ASS_ASSIGNMENT"],1)

		nrows = len(preproc_data[i].index)
		#print(str(nrows) + " rows to process.")

		#print("Grouping by dates...")
		rcvcall = preproc_data[i].groupby(["DATE"])['CSPL_RECEIVED_CALLS'].sum()
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
		m = preproc_data[i].groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"]
		m = m.transform(np.mean)
		s = preproc_data[i].groupby(["WEEKDAY"])["CSPL_RECEIVED_CALLS"].transform(np.std)
		preproc_data[i].loc[:, "WEEKDAY_MEAN"] = m
		preproc_data[i].loc[:, "WEEKDAY_STD"] = s
		#print preproc_data[i]

		mean, std = extract_mean_std(preproc_data[i])

		mean_ass.append(mean)
		std_ass.append(std)

	#print (mean_ass)
	#print(std_ass)

	return mean_ass, std_ass


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
    os.chdir("/home/nicolas/Documents/INF554 - Machine Learning/AXA Data Challenge")
    read_file_content("train_2011_2012_2013.csv", "submission.txt", 20000)