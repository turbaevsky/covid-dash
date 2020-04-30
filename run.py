#!/usr/bin/python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

#from app import app

from dash.dependencies import Input, Output, State
import os
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#%matplotlib inline
import math
import logging

#logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.DEBUG)
#logging.info('Start')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,\
routes_pathname_prefix='/covid/')
server = app.server

def load():
#    import pandas as pd
    data = pd.read_csv('daily.csv')
    country = data['Country/Region'].drop_duplicates().values

    return data, country


def logistic_model(x,a,b,c):
#    import numpy as np
    return c/(1+np.exp(-(x-b)/a))

def logistic(x,mu,s,mult):
#    import numpy as np
    return mult*(np.exp(-(x-mu)/s))/(s*(1+np.exp(-(x-mu)/s))**2)

def plot(x,y,p):
    logging.debug('population= %d' % p)
    cum = list(y)
    x = list(x)
    y = [int(y[i]-y[i-1]) if i!=0 else y[0] for i in range(0,len(y))]

    # Cumulative
    fitC = curve_fit(logistic_model,x,cum,bounds=(0,[10,200,p]),maxfev=1e5)
    errorsC = [np.sqrt(fitC[1][i][i]) for i in [0,1,2]]
    tC = '\nThe total expected number of people affected &nbsp; is %d +/- %d, so it might increase up to\n\
 %1.1f times in comparison to today`s %d'  \
    % (fitC[0][2],errorsC[2],fitC[0][2]/cum[-1], cum[-1])

    # Daily
    errors = [1e3]
    while errors[0]>10:
        pop = 0.9*p
        fit = curve_fit(logistic,x,y,bounds=(0,[2e2,10,pop]),maxfev=1e5)
        mu, sigma, mult = fit[0][0], fit[0][1], fit[0][2]
        dev3 = sigma*math.pi*math.sqrt(3)
        inflect = [mu + sigma*math.log(2-math.sqrt(3)),mu + sigma*math.log(2+math.sqrt(3))]
        #from scipy.integrate import quad
        #I = quad(logistic, mu-dev3, mu+dev3, args=(mu, sigma, mult))
        p = pop
        logging.debug('fit: %s' % str(fit))
        errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]


    pred_x = list(range(max(x)+1,int(mu+dev3))) if (mu+dev3)>max(x) else []
    try:
        start = x.index(int(mu-dev3))
    except:
        start = x[0]

    x = x[start:]
    y = y[start:]


    logging.debug('fit: %s' % str(fit))
    errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
    #a, b, c = fit[0][0], fit[0][1], fit[0][2]
    #sol = int(fsolve(lambda x : logistic(x,a,b,c) - int(c),b))
    #plt.scatter(x,y,label='%s' % title,color="red")# Predicted logistic curve    
    #var = int(math.pow(fit[0][1],2)*math.pow(math.pi,2)/3)
    #pred_x = list(range(max(x)+1,int(fit[0][0])+var)) if max(x)<(int(fit[0][0])+var) else []

    Y = [int(logistic(i,fit[0][0],fit[0][1],fit[0][2])) for i in x+pred_x]
    
    txt = 'The  peak is expected on %.10s +/- %d days  with %d people affected at the moment of the peak. ' \
% (datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(days=fit[0][0]), errors[0], max(Y))
    txt += 'The expected number of affected people \n at the given period end is %d +/- %d\n' % (fit[0][2],errors[2])
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
    html.A(children='Main Site', href='http://3.134.253.206'),
    html.Div(),
    html.A(children='Contact', href='mailto:turbaevsky@gmail.com'),
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
        df = df[(df['Country/Region']==country) & (df['Province/State']==state)]
        pop = pd.read_csv('population.csv') # population
        try:
            p = int(pop[pop['Country Name']==country]['2018'].values[0])
            #print(p)
        except Exception as e:
            print(e)
            p=1e7
        options=[{'label': i, 'value': i} for i in ['Infected', 'Recovered','Dead','Active']]
        #cn=[{'label': i, 'value': i} for i in load()[1]]
        cn, st = False, False

    else:
        # read data from gov.uk
        url = 'https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data' #daily
        daily = pd.read_excel(url, header=0)
        date = daily['DateVal']
        FMT = '%Y-%m-%d'
        daily['data'] = date.map(lambda x : (x - datetime.strptime("2020-01-01", FMT)).days)
        daily = daily.drop(columns=['CMODateCount','DailyDeaths'])
        daily.columns=['Date','Confirmed','Dead','data']
        df = daily.fillna(0)
        p = 67e6
        options=[{'label': i, 'value': i} for i in ['Infected', 'Dead']]
        cn, st = True, True

    X = df.data.apply(int).values
    Y = df.Active.values if display=='Active' else df.Confirmed.values if display=='Infected' else \
        df.Recovered.values if display=='Recovered' else df.Dead.values

    _, yy, YY, xx, text  = plot(X,Y,p)

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

    return figure, text, p, options, cn, st


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
