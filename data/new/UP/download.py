import time
import datetime
import os
import csv
from selenium import webdriver
#from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

months1 = ["January","February","March","April","May","June","July","August","September","October","November","December"]
months2 = ["January","February","March","April","May","June","July","August","September","October","November"]
months3 = ["December"]

'''
start_year = 2014
end_year = 2017
mandi_file = pd.read_csv('mandis.csv')
mandicode = mandi_file['mandicode']
mandiname = mandi_file['mandiname']
mandistate = mandi_file['statecode']
mandi_map = {}
mandi_state_map={}
i=0
for row in mandiname:
	mandi_map[row] = mandicode[i]
	mandi_state_map[row] = mandistate[i]
	i = i+1
'''

centernames = ["Uttar Pradesh"]

def extractdata():

	path_to_chromedriver = '/home/praneet/Downloads/chromedriver' 
	browser = webdriver.Chrome(executable_path = path_to_chromedriver)

	# browser = webdriver.Chrome()
	url = 'http://agmarknet.nic.in/agnew/NationalBEnglish/DatewiseCommodityReport.aspx'
	browser.get(url)
	print "1"
	myfile= open('mynewdata.csv','a')
	for center in centernames:
		start_year = 2011
		end_year = 2017
		for year in range(start_year,end_year+1):
			months = months1
			if(year == 2017):
				months = months2
			# elif(year == 2016):
			# 	months = months3
			
			for month in months:
				print year,month
				# browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"cboYear\"]/option[contains(text(),\""+str(year)+"\")]").click()
				browser.find_element_by_xpath("//*[@id=\"cboMonth\"]/option[contains(text(),\""+month+"\")]").click()
				# browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"cboState\"]/option[contains(text(),\""+center+"\")]").click()
				# browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"cboCommodity\"]/option[contains(text(),\""+"Onion"+"\")]").click()
				# browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"btnSubmit\"]").click()
				table = browser.find_element_by_xpath("//*[@id=\"gridRecords\"]")
				rows = table.find_elements_by_tag_name("tr")
				count = 0
				for row in rows:
					cells = row.find_elements_by_xpath(".//*[local-name(.)='th' or local-name(.)='td']")
					st = ''
					for cell in cells:
						st +=cell.text+','
					st+='\n'
					myfile= open('mynewdata_'+str(year)+'_'+str(month)+"_"+str(center)+'.csv','a')
					myfile.write(st)
					myfile.close()
				browser.find_element_by_xpath("//*[@id=\"LinkButton1\"]").click()

if __name__ == '__main__':
	extractdata()
	