# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:52:41 2020

@author: batesc

Takes a .csv of hourly feeder loading data and projects it out for max and min 
values for a year. To be ran in a time-series load flow analysis.

"""


import pandas as pd
from fbprophet import Prophet

# instantiate the model and set parameters
model = Prophet(
    changepoint_prior_scale=0.01,
    interval_width=0.95,
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=True,
    seasonality_mode='additive'
)

history_pd = pd.read_csv("load.csv")

# fit the model to historical data
model.fit(history_pd)

# projects over 8760 hours - 1 year
future_pd = model.make_future_dataframe(
    periods=8760,
    freq='H',
    include_history=True
)

# predict over the dataset
forecast_pd = model.predict(future_pd)

fig1 = model.plot(forecast_pd, xlabel='date', ylabel='load')
fig2 = model.plot_components(forecast_pd)
