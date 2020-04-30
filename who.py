#!/usr/bin/python3

from datetime import datetime
from tqdm import tqdm
import logging
import sys

logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def pivot(df,cols=4):
    
    import pandas as pd

    l = len(df.iloc[0])-4
    col = df.columns
    d = pd.DataFrame()
    for i in tqdm(range(len(df))):
        #print('.',end='')
        #print(df.iloc[i])
        for j in range(l):
            #print(df.iloc[i][j], type(df.iloc[i][j]))
            dd = pd.DataFrame(data={'Province/State':[df.iloc[i][0]],'Country/Region':[df.iloc[i][1]],\
                      'Lat':[df.iloc[i][2]],'Long':[df.iloc[i][3]],'Date':[col[j+cols]],\
                      'value':[df.iloc[i][j+cols]]})
            #print(dd)
            d = d.append(dd, ignore_index=True)
            #print(d)
    d['Date'] = d.Date.apply(lambda x:(datetime.strptime(x,'%m/%d/%y').strftime("%Y-%m-%d")))
    return d


def main():
	# TODO Add check whether file been modified checking md5
	import pandas as pd

	confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
	dead = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
	recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

	ch = str(pd.util.hash_pandas_object(confirmed).sum())
	dh = str(pd.util.hash_pandas_object(dead).sum())
	rh = str(pd.util.hash_pandas_object(recovered).sum())

	try:
		with open('hash.txt','r') as f:
			content = f.readlines()
			content = [x.strip() for x in content]
			logging.debug(content)
			if ch == content[0] and dh == content[1] and rh == content[2]:
				logging.info('Nothing to update')
				sys.exit(1)
			else:
				with open('hash.txt','w') as f:
					f.write('%s\n' % ch)
					f.write('%s\n' % dh)
					f.write('%s\n' % rh)
	except Exception as e:
		with open('hash.txt','w') as f:
			logging.warning(e)
			f.write('%s\n' % ch)
			f.write('%s\n' % dh)
			f.write('%s\n' % rh)



	confirmed = pivot(confirmed)
	#print('*',end='')
	dead = pivot(dead)
	#print('*',end='')
	recovered = pivot(recovered)

	confirmed.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Confirmed']
	dead.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Dead']
	recovered.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Recovered']

	data = confirmed.merge(recovered, on=['Date','Province/State','Country/Region'], how='left').drop(columns=['Lat_y','Long_y','Lat_x','Long_x'])
	data = data.merge(dead, on=['Date','Province/State','Country/Region'], how='left')#.drop(columns=['Lat','Long'])
	data.fillna({'Province/State':'main','Dead':0,'Active':0,'Recovered':0,'Confirmed':0},inplace=True)
	data['Active'] = data.Confirmed-data.Recovered-data.Dead
	data['data'] = data.Date.map(lambda x : (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2020-01-01", "%Y-%m-%d")).days)

	data.to_csv('daily.csv')


if __name__ == '__main__':
	main()
