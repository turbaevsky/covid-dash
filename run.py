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

def plot(x,y):
#    import numpy as np
    cum = list(y)
    x = list(x)
    pred_x = list(range(max(x)+1,140))
    y = [int(y[i]-y[i-1]) if i!=0 else y[0] for i in range(0,len(y))]
    #upd = str(daily.tail(1).DateVal.values[0])[:10]
    #txt = 'Data updated on %s\n\n' % upd

    # Cumulative
    #fitC = curve_fit(logistic_model,x,cum,p0=[2,130,2e6])
    fitC = curve_fit(logistic_model,x,cum,bounds=(0,[10,200,1e7]),maxfev=1e5)

    errorsC = [np.sqrt(fitC[1][i][i]) for i in [0,1,2]]
    tC = '\nThe total expected number of people affected &nbsp; is %d +/- %d, so it might increase up to\n\
 %1.1f times in comparison to today`s %d'  \
    % (fitC[0][2],errorsC[2],fitC[0][2]/cum[-1], cum[-1])

    #fit = curve_fit(logistic,x,y,p0=[100,5,1e5])
    fit = curve_fit(logistic,x,y,bounds=(0,[2e2,10,1e7]),maxfev=1e5)
    #print(fit)
    errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
    a, b, c = fit[0][0], fit[0][1], fit[0][2]
    sol = int(fsolve(lambda x : logistic(x,a,b,c) - int(c),b))
    #plt.scatter(x,y,label='%s' % title,color="red")# Predicted logistic curve    

    Y = [int(logistic(i,fit[0][0],fit[0][1],fit[0][2])) for i in x+pred_x]
    
    txt = 'The  peak is expected on %.10s +/- %d days  with %d people affected. ' \
% (datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(days=fit[0][0]), errors[0], max(Y))
    txt += 'The expected number of affected people \n at period end is %d +/- %d\n' % (fit[0][2],errors[2])
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
    html.H1(children='COVID Forecast'),

    html.Div(children='''
        We use ONLY official data from gov.uk and who.org. 
        The model is autotuned, so day by day data might be different. 
        The model is updated every midnight.
    '''),

        html.Div([
        dcc.Dropdown(
            id='country',
            options=[{'label': i, 'value': i} for i in load()[1]],
            value='United Kingdom'
        ),

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
            number of recovered and dead ones''')
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
            Output('txt','children')],
            [Input('country','value'),
             Input('state','value'),
             Input('display','value')])
def chart(country,state,display):
    df=load()[0]
    df = df[(df['Country/Region']==country) & (df['Province/State']==state)]
    X = df.data.apply(int).values
    Y = df.Active.values if display=='Active' else df.Confirmed.values if display=='Infected' else \
        df.Recovered.values if display=='Recovered' else df.Dead.values

    _, yy, YY, xx, text  = plot(X,Y)

    figure={
    'data': [
        {'x': xx, 'y': yy, 'type': 'bar', 'name': 'Real data'},
        {'x': xx, 'y': YY, 'type': 'line', 'name': 'Forecast'},
    ],
    'layout': {
        'title': 'Daily %s People' % display
        }
        }
    return figure, text


if __name__ == '__main__':
    app.run_server(debug=True)
