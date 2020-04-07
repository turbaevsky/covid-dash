#!/usr/bin/python3

import pandas as pd

def load():

	u = 'https://www.arcgis.com/sharing/rest/content/items/bc8ee90225644ef7a6f4dd1b13ea1d67/data' #daily
	url = 'https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data' #daily stat
	url2 = 'https://www.arcgis.com/sharing/rest/content/items/ca796627a2294c51926865748c4a56e8/data' #regions
	url3 = 'https://www.arcgis.com/sharing/rest/content/items/b684319181f94875a6879bbc833ca3a6/data' #county

	daily = pd.read_excel(url, header=0).to_csv('daily.csv')
	reg = pd.read_csv(url2, header=0).to_csv('reg.csv')
	county = pd.read_csv(url3, header=0).to_csv('county.csv')
	snap = pd.read_excel(u, header=0).to_csv('snap.csv')

	return 0


if __name__ == '__main__':
	load()