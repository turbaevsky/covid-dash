#!/usr/bin/python3

from datetime import datetime
import logging
import sys
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

'''
def pivot(df):
    df = df.fillna({'Province/State':'main'}).fillna(0)
    df = df.pivot_table(columns=['Province/State','Country/Region','Lat','Long'])
    df = df.reset_index()
    logging.debug(df.columns())
    logging.debug(df.tail())
    df['Date'] = df.level_0.apply(lambda x:(datetime.strptime(x,'%m/%d/%y').strftime("%Y-%m-%d")))
    df.drop(columns=['level_0'], inplace=True)
    df.columns = ['Province/State','Country/Region','Lat','Long','value','Date']
    return df
'''

def pivot(ddf):
    ddf = ddf.fillna({'Province/State':'main'}).fillna(0)
    col = ddf.columns[4:].tolist()
    d = {'Province/State':[], 'Country/Region':[], 'Lat':[], 'Long':[], 'Value':[], 'Date':[]}
    d = pd.DataFrame(d)
    for n in range(len(ddf.index)):
        print(n, end='\x1b[1K\r')
        for i in range(len(col)):
            pos = 4 + i
            #yr = int(col[i][:4])
            #val = float('{:.1f}'.format(float(ddf.iloc[n,pos]))) if ddf.iloc[n,pos] != '..' else 0
            # val might be replace with None then dropna()
            dt = datetime.strptime(col[i],'%m/%d/%y').strftime("%Y-%m-%d")
            ln = {'Province/State':[ddf.iloc[n,0]],
                  'Country/Region':[ddf.iloc[n,1]],
                  'Lat':[ddf.iloc[n,2]],
                  'Long':[ddf.iloc[n,3]],
                  'Value':[ddf.iloc[n,pos]],
                  'Date':[dt]
                    }
            l = pd.DataFrame(ln)
            d = d.append(l, ignore_index=True)
    return d


def main():
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
			if ch == content[0] and dh == content[1] and rh == content[2]: ################
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

	confirmed.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Confirmed', 'Date']
	dead.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Dead', 'Date']
	recovered.columns = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Recovered', 'Date']

	data = confirmed.merge(recovered, on=['Date','Province/State','Country/Region'], how='left').drop(columns=['Lat_y','Long_y','Lat_x','Long_x'])
	data = data.merge(dead, on=['Date','Province/State','Country/Region'], how='left')#.drop(columns=['Lat','Long'])
	#data.fillna({'Province/State':'main','Dead':0,'Active':0,'Recovered':0,'Confirmed':0},inplace=True)
	data['Active'] = data.Confirmed-data.Recovered-data.Dead
	data['data'] = data.Date.map(lambda x : (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2020-01-01", "%Y-%m-%d")).days)

	data.to_csv('daily.csv')
	logging.info('File updated')


if __name__ == '__main__':
	main()
