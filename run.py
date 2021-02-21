#!/usr/bin/python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

#from app import app

from dash.dependencies import Input, Output, State
import os
from pandas import read_csv
from numpy import exp, sqrt
from datetime import datetime,timedelta,date
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#%matplotlib inline
import math
import logging
import memory_profiler

import pandas as pd
import numpy as np

m1 = memory_profiler.memory_usage()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,\
routes_pathname_prefix='/covid/')
server = app.server

def load():
#    import pandas as pd
    data = read_csv('daily.csv')
    country = data['Country/Region'].drop_duplicates().values

    return data, country


def logistic_model(x,a,b,c):
#    import numpy as np
    return c/(1+exp(-(x-b)/a))

def logistic(x,mu,s,mult):
# to fit a curve
    return mult*(exp(-(x-mu)/s))/(s*(1+exp(-(x-mu)/s))**2)

def ord(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f


def plot(x,y,p):
    logging.debug('population= %d' % p)
    cum = list(y)
    x = list(x)
    y = [int(y[i]-y[i-1]) if i!=0 else y[0] for i in range(0,len(y))]
    y = [a if a>0 else 0 for a in y]
    
    x = x[1:]
    y = y[1:]
    cum = cum[1:]
    
    logging.debug('x input for plot is %s' % str(x))
    logging.debug('y input for plot is %s' % str(y))
    logging.debug('x len is {}, y len is {}'.format(len(x), len(y)))
    # Cumulative
    fitC = curve_fit(logistic_model,x,cum,bounds=(0,[10,200,p]),maxfev=1e5)
    errorsC = [sqrt(fitC[1][i][i]) for i in [0,1,2]]
    tC = '\nThe total expected number of people affected &nbsp; is %d +/- %d, so it might increase up to\n\
 %1.1f times in comparison to today`s %d'  \
    % (fitC[0][2],errorsC[2],fitC[0][2]/cum[-1], cum[-1])
    logging.debug('Cummulative fit completed')


    # Daily
    errors = [1e3]
    while errors[0]>10:
        pop = 0.9*p
        fit = curve_fit(logistic,x,y,bounds=(0,[4e2,1e2,pop]),maxfev=1e6)
#        fit = curve_fit(ord, x, y, maxfev=1e5)
        logging.debug('\nFIT ====> {}\n'.format(fit))

        mu, sigma, mult = fit[0][0], fit[0][1], fit[0][2]
        dev3 = sigma*math.pi*math.sqrt(3)
        inflect = [mu + sigma*math.log(2-math.sqrt(3)),mu + sigma*math.log(2+math.sqrt(3))]
        #from scipy.integrate import quad
        #I = quad(logistic, mu-dev3, mu+dev3, args=(mu, sigma, mult))
        p = pop
        #logging.debug('fit: %s' % str(fit))
        errors = [sqrt(fit[1][i][i]) for i in [0,1,2]]


    pred_x = list(range(max(x)+1,int(mu+dev3))) if (mu+dev3)>max(x) else []
    try:
        start = x.index(int(mu-dev3))
    except:
        start = 0
    logging.debug('start position is %s' % start)

    x = x[start:]
    y = y[start:]
    #logging.debug('y output for plot is %s' % str(y))

    logging.debug('fit: %s' % str(fit))
    errors = [sqrt(fit[1][i][i]) for i in [0,1,2]]
    #a, b, c = fit[0][0], fit[0][1], fit[0][2]
    #sol = int(fsolve(lambda x : logistic(x,a,b,c) - int(c),b))
    #plt.scatter(x,y,label='%s' % title,color="red")# Predicted logistic curve    
    #var = int(math.pow(fit[0][1],2)*math.pow(math.pi,2)/3)
    #pred_x = list(range(max(x)+1,int(fit[0][0])+var)) if max(x)<(int(fit[0][0])+var) else []

    Y = [int(logistic(i,fit[0][0],fit[0][1],fit[0][2])) for i in x+pred_x]
    
    txt = ''
    txt = 'The peak is expected on %.10s +/- %d days  with %d people affected at the moment of the peak. ' \
% (datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(days=fit[0][0]), errors[0], max(Y))
    txt += '\nThe end of period (when 99.7 percent will be covered by curve) is expected on %.10s +/- %d days.' \
    % (datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(days=(mu+dev3)), errors[0])
    txt += '\nThe expected number of affected people \n at the given period end is %d +/- %d\n' % (fit[0][2],errors[2])
    #txt += 'The expected end is %.10s\n' % (datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(days=fit[0][0]+fit[0][1]**2*math.pi**2/3))

    #last = y.tail(1).CMODateCount.values[0]
    #mult = max(Y) / last
    #txt += ', which indicates\n an increase of up to %1.1f times\n in comparison to today`s %d\n' % (mult,last)
    #txt += tC

    ticks = [datetime.strftime(datetime.strptime("2020-01-01","%Y-%m-%d") + timedelta(days=int(x)),"%d %b") \
                     #if int(x/5)==x/5 else None 
                     for x in x+pred_x]

    return x+pred_x, y, Y, ticks, txt


app.layout = html.Div(children=[
    html.A(children='Main Site', href='https://livedata.link'),
    html.Div(),
    html.A(children='Contact', href='mailto:ubuntu@livedata.link'),
    html.H1(children='COVID Forecast'),

    html.Div(children='''
        We use ONLY official data from who.org or gov.uk. 
        The model is autotuned, so day by day data might be different. 
        The model is updated every midnight.
    '''),
        dcc.RadioItems(id='radio',options=[
        {'label': 'WHO', 'value': 'who'},
        {'label': 'UK Government', 'value': 'uk', 'disabled': True},
    ],
    value='who',
    labelStyle={'display': 'inline-block'}
    ),
        html.Div([
            html.Div(children='Please select the country'),
        dcc.Dropdown(
            id='country',
            options=[{'label': i, 'value': i} for i in load()[1]],
            value='United Kingdom'
        ),
        html.Div(children='Please select the state or region'),
        dcc.Dropdown(
            id='state',
            options=[{'label': 'main', 'value': 'main'}],
            value='main'
        ),

        dcc.RadioItems(
            id='display',
            options=[{'label': i, 'value': i} for i in ['Infected', 'Recovered','Dead','Active',]],
            value='Infected',
            labelStyle={'display': 'inline-block'}
        ),

        dcc.Markdown(''' The Active means the number of infected people minus the 
            number of recovered and dead ones'''),
        html.Div(id='pop',children='population'),
    ]),

    html.H5(id='txt'),

    dcc.Graph(
        id='graph',
        ),
])


@app.callback(Output('state','options'),
            [Input('country','value')])
def st(country):
    df=load()[0]
    lst = df[df['Country/Region']==country]['Province/State'].drop_duplicates().values
    logging.debug('Countries = %s' % lst)
    options=[{'label': i, 'value': i} for i in lst]
    return options or None


@app.callback([Output('graph','figure'),
            Output('txt','children'),
            Output('pop','children'),
            Output('display','options'),
            Output('country','disabled'),
            Output('state','disabled')],
            [Input('radio','value'),
             Input('country','value'),
             Input('state','value'),
             Input('display','value')])
def chart(radio,country,state,display):
    if radio=='who':
        df=load()[0]
        
        startDate = np.datetime64(date(2020,10,1))
        #print(df.tail())
        #print(df.columns)
        
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        
        df = df[(df['Country/Region']==country) & 
			(df['Province/State']==state) &
			(df['Date']>startDate)]
        logging.debug('df = %s' % df)
        pop = read_csv('population.csv') # population
        try:
            p = int(pop[pop['Country Name']==country]['2018'].values[0])
            #print(p)
        except Exception as e:
            print(e)
            p=1e7
        options=[{'label': i, 'value': i} for i in ['Infected', 'Recovered','Dead','Active']]
        #cn=[{'label': i, 'value': i} for i in load()[1]]
        cn, st = False, False

    X = df.data.apply(int).values
    Y = df.Active.values if display=='Active' else df.Confirmed.values if display=='Infected' else \
        df.Recovered.values if display=='Recovered' else df.Dead.values

    logging.debug('Y = %s' % Y)

    _, yy, YY, xx, text  = plot(X,Y,p)  

    logging.debug('y for plotting is %s' % yy)

    figure={
    'data': [
        {'x': xx, 'y': yy, 'type': 'bar', 'name': 'Real data'},
        {'x': xx, 'y': YY, 'type': 'line', 'name': 'Forecast'},
    ],
    'layout': {
        'title': 'Interactive Chart for Daily %s Cases in %s' % (display, country)
        }
        }

    p = 'The whole population of the country is %d' % p

    m2 = memory_profiler.memory_usage()

    logging.debug('mem usage %d, %d diff is %d Mb' % (m1[0], m2[0], m2[0]-m1[0]))

    return figure, text, p, options, cn, st


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
