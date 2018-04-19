from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing
import os

from averagemandi import getmandi2

dil = pd.read_excel('data/DELHImandis.xlsx')
mum = pd.read_excel('data/MUMBAImandis.xlsx')
luck = pd.read_excel('data/LUCKNOWmandis.xlsx')

def findeligiblemandis(df1):
	for i in range(0,len(df1)):
		df = df1[df1['ratio'] >= 0.5]
	return list(df['name'])

delhi = [getmandi2(m,False) for m in findeligiblemandis(dil)]
mumbai = [getmandi2(m,False) for m in findeligiblemandis(mum)]
lucknow = [getmandi2(m,False) for m in findeligiblemandis(luck)]


def findratio(listofseries):
	t = []
	for l in listofseries:
		l1 = l[l!=0]
		t.append(l1.groupby(l1.index.month).mean())
	m = np.array(t)
	# m is mandis vs month
	for i in range(0,12):
		m[:,i] = m[:,i]/m[:,i].sum()
	return m 

def getcombinedseries(listofseries,m):
	df = pd.concat(listofseries, axis=1)
	ans = pd.Series(0.0,index=df.index)
	for i in range(1,13): 
		w = m[:,i-1]
		t = df[df.index.month == i]
		for index, row in t.iterrows():
			s1=0
			w1=0
			aperw = 0
			r = list(row)
			for j in range(len(w)):
				if(r[j]!=0):
					s1 = s1+r[j]
					w1 = w1+w[j]
			if(w1!=0):
				aperw = s1/w1
				for j in range(len(w)):
					if(r[j]==0):
						r[j] = w[j]*aperw
			ans[index] = sum([w[ind]*r[ind] for ind in range(len(w))])
	ans = (ans.replace(0.0, np.nan)).interpolate(method='pchip',limit_direction='both')
	return ans


delhians = getcombinedseries(delhi,findratio(delhi))
mumbaians = getcombinedseries(delhi,findratio(delhi))
lucknowans = getcombinedseries(delhi,findratio(delhi))