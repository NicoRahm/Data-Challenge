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


def updateSub(prediction, filename_ori = "submission.txt", filename_pred = "submission_with_pred.txt"):

	with open(filename_pred, "w", encoding = 'UTF-8') as fichier:

		fichier.write("DATE\tASS_ASSIGNMENT\tprediction\r")

		with open(filename_ori, "r") as original:
			
			content = original.readlines()
			start = True

			for line in content:
				if (start):
					start = False
				else:

					splitted = line.split("	")
					date = extract_date(splitted[0])
					assignment = splitted [1]

					pred = prediction.loc[assignment].loc[date]
#					pred = int(pred)

					#test = 8.0
					#test = int(test)

					fichier.write(str(date) + ".000\t" + assignment + "\t" + str(pred) + "\r")
	return()
def extract_date(string):
    d = dt.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.000")
    return(d)



if __name__ == '__main__':
	updateSub("test", "submission.txt")