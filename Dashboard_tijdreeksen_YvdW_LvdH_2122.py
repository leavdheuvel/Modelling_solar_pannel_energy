# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 20:35:32 2022

@author: Lea

Dashboard tijdreeksen 2122
Ymke van der Waal en Lea van den Heuvel (18057020)

Dit is de code voor de dashboard behorende bij de eindopdracht.

Om de app te runnen:
1. Run dit bestand in prompt (python filename.py)
2. Ga naar http://127.0.0.1:8050/

"""

#%% Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import dateparser
from datetime import timedelta
import pmdarima as pm
from statsmodels.graphics.gofplots import qqplot
import datetime
from scipy.stats import shapiro
from statsmodels.tsa.stattools import acf, pacf

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

# plt.rcParams["figure.figsize"] = (15,8)
# plt.rcParams['image.cmap'] = 'Paired'
np.random.seed(42)

#%% Functions for plotting
def plot_rolling_mean(df, df_rolling, col, title):
    """
    Returns plot met daarin ogirinele tijdreeks en rolling mean
    Input:
        
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index,
                                        y=df[col],
                                        name='Meting',
                                        line_color='rgb(0,204,204)'
                                       ))

    fig.add_trace(go.Scatter(x=df_rolling.index,
                                        y=df_rolling[col],
                                        name="Rolling mean (7 dagen)",
                                        line_color='rgb(64,64,64)'
                                       ))

    fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right",
                                          x=1),
                                 title={
                                     'text': title,
                                     'y':0.86,
                                     'x': 0.08},
                                 template="ggplot2"
                                )

    return fig

def plot_acf(values_acf, fig, row, col):
    """
    Returns figuur met acf, zonder conf int
    Input is acf- en pacf-waarden, en titel van de figuur
    """
    
    # Plot acf
    for i in range(len(values_acf)):
        fig.add_trace(go.Scatter(x=[i+1,i+1], 
                                 y=[0,values_acf[i]],
                                 line_color = 'rgb(248,118,109)'),
                      row=row, col=col
                     )

    return fig

def plot_pacf(values_pacf, fig, row, col):
    """
    Returns figuur met acf en pacf, zonder conf int
    Input is acf- en pacf-waarden, en titel van de figuur
    """
    for i in range(len(values_pacf)):
        fig.add_trace(go.Scatter(x=[i+1,i+1], 
                                 y=[0,values_pacf[i]],
                                 line_color = 'rgb(248,118,109)'
                                ),
                      row=row, col=col)
    return fig

def gen_values_acf(series):
    """
    Returns acf waarden en conf int van kolom
    Input is dataframe en kolom van desbetreffende tijdreeks
    """
    acf_x, acf_confint = acf(series,
                     nlags=10,
                     alpha=0.05,
                     fft=False,
                     bartlett_confint=True
                    )
    return acf_x[1:], acf_confint[1:]

def gen_values_pacf(series):
    """
    Returns pacf waarden en conf int van kolom
    Input is dataframe en kolom van desbetreffende tijdreeks
    """
    pacf_x, pacf_confint = pacf(series,
                                nlags=10,
                                alpha=0.05,
                                method = 'yw'
                               )
    return pacf_x[1:], pacf_confint[1:]

def gen_lags(n_lags):
    """
    Returns de x-coördinaten voor acf en pacf
    Input is aantal lags
    """
    lags = np.arange(1, n_lags).astype('float')
    lags[0] -= 0.5
    lags[-1] += 0.5
    return lags

def plot_anom_result(df, cols, anom_col, fig, row, col):
    """
    Returned plot met timeseries en anomalies aangekaart
    
    ts: dataframe
    cols: str, kolomnamen waarop anomalies gebaseerd zijn
    anom_col: str, kolomnaam waarin staat aangegeven welke punten anomalies zijn
    fig: plotly figuur waarin geplot wordt
    row: int, rij van subplots waarin geplot moet worden
    col: int, kolom van subplots waarin geplot moet worden
    """
    
    # Aparte ts voor anomalies
    ts_anom = df[df[anom_col] == -1]
    
    for col_name in cols:
        # Plot tijdreeksen
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df[col_name], 
                                ),
                      row=row, col=col)

        # Plot anomalies
        fig.add_trace(go.Scatter(x=ts_anom.index,
                                 y=ts_anom[col_name],
                                 mode='markers',
                                 marker_color='red'),
                      row=row, col=col)
    
    return fig

#%% Read data and set date_time index
ts = pd.read_csv('Data_tijdreeksen.csv', index_col = 'date')
ts.index = pd.to_datetime(ts.index)
print(ts.info())

#%% Rolling mean plot
ts_rolling = ts.rolling(window = 7, center = True).mean().dropna()

fig_rolling_tl = plot_rolling_mean(ts, ts_rolling, 'teruglevering',
                                   'Electriciteit teruglevering op basis van zonnepanelen in kWh')
fig_rolling_tl.update_layout(yaxis_title="Waarde (kWh)",
                             xaxis_title="Datum")
# fig_rolling_tl.show()

#%% Boxplots per month
fig_boxplot = go.Figure()

fig_boxplot.add_trace(go.Box(x=ts["month"], y=ts["teruglevering"],
                             marker_color='rgb(199, 124, 255)',
                             line_color='rgb(248,118,109)'
                            ))

# Labels en plaatsing labels
fig_boxplot.update_layout(title={
                             'text': "Boxplots voor elektriciteit teruglevering per maand",
                             'y':0.88,
                             'x': 0.06},
                          margin={"r":0,"l":0,"b":0},
                          yaxis_title="Elektriciteit teruglevering (kWh)",
                          template="ggplot2",
                          width = 800, height = 600,
                          xaxis = dict(tickmode = 'array',
                                       tickvals = [*range(1, 13)],
                                       ticktext = ['Jan', 'Feb', 'Maart', 'April', 'Mei', 'Juni', 'Juli',
                                                   'Aug','Sept', 'Okt', 'Nov', "Dec"]
    )
)

# fig_boxplot.show()

#%% Histogram per season
# Twee verschillende
ts_spring = ts[ts['season']=="Spring"]
ts_summer = ts[ts['season']=="Summer"]
ts_autumn = ts[ts['season']=="Autumn"]
ts_winter = ts[ts['season']=="Winter"]

fig_hist1 = make_subplots(rows=2, cols=2, start_cell="top-left")

fig_hist1.add_trace(go.Histogram(x=ts_spring['teruglevering'], name = 'Lente'),
                  row=1, col=1)

fig_hist1.add_trace(go.Histogram(x=ts_summer['teruglevering'], name = "Zomer"),
                  row=1, col=2)

fig_hist1.add_trace(go.Histogram(x=ts_autumn['teruglevering'], name = 'Herfst'),
                  row=2, col=1)

fig_hist1.add_trace(go.Histogram(x=ts_winter['teruglevering'], name = 'Winter'),
                  row=2, col=2)

fig_hist1.update_layout(title={
                             'text': "Histogrammen elektriciteit teruglevering per seizoen",
                             'y':0.88,
                             'x': 0.06},
                       template="ggplot2")

# fig_hist1.show()

# Of deze?
fig_hist2 = go.Figure()

dict_bins = dict(start=0, end=30, size=3)

fig_hist2.add_trace(go.Histogram(x=ts_spring['teruglevering'], 
                           name = 'Lente', 
                           xbins=dict_bins))

fig_hist2.add_trace(go.Histogram(x=ts_summer['teruglevering'], 
                           name = "Zomer", 
                          xbins=dict_bins))

fig_hist2.add_trace(go.Histogram(x=ts_autumn['teruglevering'], 
                           name = 'Herfst', 
                          xbins=dict_bins))

fig_hist2.add_trace(go.Histogram(x=ts_winter['teruglevering'], 
                           name = 'Winter', 
                          xbins=dict_bins))

# Overlay both histograms
fig_hist2.update_layout(barmode='overlay',
                        title={
                             'text': "Histogrammen elektriciteit teruglevering per seizoen",
                             'y':0.88,
                             'x': 0.06},
                        template="ggplot2"
                       )
# Reduce opacity to see both histograms
fig_hist2.update_traces(opacity=0.5)
# fig_hist2.show()

#%% Means per dayof the week, day of the month and month
ts_grouped_day = ts.groupby(['day_of_week', 'day_of_week_name'])['teruglevering']\
                                                                                .mean()\
                                                                                    .reset_index()
ts_grouped_month = ts.groupby('day_of_month')['teruglevering'].mean()

ts_grouped_year = ts.groupby(['month', 'month_name'])['teruglevering']\
                                                                    .mean()\
                                                                            .reset_index()
                                                                            
                                                                            # Callback toevoegen!
fig_mean = px.line(ts_grouped_day, x="day_of_week", y="teruglevering", 
                   title='Gemiddeld elektriciteit teruglevering')

fig_mean.update_layout(template="ggplot2"
                       )
# fig_mean.show()

#%% ACF's and PACF's of original
acf_x, acf_confint = gen_values_acf(ts['teruglevering'])
pacf_x, pacf_confint = gen_values_pacf(ts['teruglevering'])
lags = gen_lags(11)

fig_corr1 = make_subplots(rows=2, cols=1)

# Plot acf en pacf
plot_acf(acf_x, fig_corr1, 1, 1)
plot_pacf(pacf_x, fig_corr1, 2, 1)

#Conf int acf
fig_corr1.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1] - acf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
             row = 1, col=1)

fig_corr1.add_trace(go.Scatter(x=lags,y=acf_confint[:, 0] - acf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_corr1.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1] - pacf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr1.add_trace(go.Scatter(x=lags,y=pacf_confint[:, 0] - pacf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr1.update_layout(title={'text':'ACF en PACF voor elektriciteit teruglevering'})

# fig_corr1.show()

#%% ACF and PACF of diff
teruglevering_diff = ts['teruglevering'].diff().dropna()
acf_x_diff, acf_confint_diff = gen_values_acf(teruglevering_diff)
pacf_x_diff, pacf_confint_diff = gen_values_pacf(teruglevering_diff)

# Plot acf en pacf
fig_corr2 = make_subplots(rows=2, cols=1)

# Plot acf en pacf
plot_acf(acf_x_diff, fig_corr2, 1, 1)
plot_pacf(pacf_x_diff, fig_corr2, 2, 1)

#Conf int acf
fig_corr2.add_trace(go.Scatter(x=lags, y=acf_confint_diff[:, 1] - acf_x_diff, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
             row = 1, col=1)

fig_corr2.add_trace(go.Scatter(x=lags,y=acf_confint_diff[:, 0] - acf_x_diff, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_corr2.add_trace(go.Scatter(x=lags, y=pacf_confint_diff[:, 1] - pacf_x_diff, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr2.add_trace(go.Scatter(x=lags,y=pacf_confint_diff[:, 0] - pacf_x_diff, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr2.update_layout(title={'text':'ACF en PACF voor gedifferentieerde elektriciteit teruglevering'})

# fig_corr2.show()

#%% Rolling mean weather data
# Callback toevoegen!
fig_rolling_zon = plot_rolling_mean(ts, ts_rolling, 'zonneschijn',
                                   'Zonneschijnduur in 0.1 uur')

fig_rolling_zon.update_layout(yaxis_title="Waarde (0.1 uur)",
                              xaxis_title="Datum")

# fig_rolling_zon.show()

#%% Correlation heatmap
# Nummerieke kolommen
cols = ['teruglevering', 
        'temp', 'zonneschijn','duur_neerslag', 'som_neerslag']

# Correlatie van teruglevering met de andere variabelen, gesorteerd van positief naar negatief
ts_corr = ts[cols].corr()
corr_teruglevering = ts_corr.drop('teruglevering')[['teruglevering']].sort_values('teruglevering', ascending = False)

fig_corr = px.imshow(np.round(corr_teruglevering, decimals = 2), # Rond af op twee cijfers achter de komma
                aspect="auto", # Maak breder figuur
                text_auto=True) # Zet corrcoeff erbij
#fig_corr.show()

#%% Regplot for weather data vs teruglevering
# Callback toevoegen voor andere weer-data!
fig_reg = px.scatter(ts, x="temp", y="teruglevering", trendline="ols")
fig_reg.update_layout(title={'text': "Teruglevering (kWh) vs temperatuur (0.1 ℃)"},
                        template="ggplot2"
                       )
# fig_reg.show()

#%% Visualise anomalies
fig_anom = make_subplots(rows=3, cols=1)

plot_anom_result(ts, ['teruglevering_sc'], 'anom_if_1', fig_anom, 1, 1)
plot_anom_result(ts, ['teruglevering_sc', 'zonneschijn_sc'], 'anom_if_1', fig_anom, 2, 1)
plot_anom_result(ts, ['teruglevering_sc', 'zonneschijn_sc', 'temp_sc'], 'anom_if_1', fig_anom, 3, 1)

# fig_test.show()

#%% ACF for original vs interpolated
acf_x_int, acf_confint_int = gen_values_acf(ts['teruglevering_int_1'])

fig_acf_int = make_subplots(rows=2, cols=1)

plot_acf(acf_x, fig_acf_int, 1, 1)
plot_acf(acf_x_int, fig_acf_int, 2, 1)

#Conf int acf
fig_acf_int.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1] - acf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
             row = 1, col=1)

fig_acf_int.add_trace(go.Scatter(x=lags,y=acf_confint[:, 0] - acf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_acf_int.add_trace(go.Scatter(x=lags, y=acf_confint_int[:, 1] - acf_x_int, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_acf_int.add_trace(go.Scatter(x=lags,y=acf_confint_int[:, 0] - acf_x_int, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_acf_int.update_layout(title=dict(text='ACF van originele en geïnterpoleerde tijdreeks'),
                          template="ggplot2")

# fig_acf_int.show()

#%% PACF for original vs interpolated
pacf_x_int, pacf_confint_int = gen_values_pacf(ts['teruglevering_int_1'])

fig_pacf_int = make_subplots(rows=2, cols=1)

plot_pacf(pacf_x, fig_pacf_int, 1, 1)
plot_pacf(pacf_x_int, fig_pacf_int, 2, 1)

#Conf int acf
fig_pacf_int.add_trace(go.Scatter(x=lags, 
                                  y=pacf_confint[:, 1] - pacf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 1, col=1)

fig_pacf_int.add_trace(go.Scatter(x=lags,y=pacf_confint[:, 0] - pacf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_pacf_int.add_trace(go.Scatter(x=lags, y=pacf_confint_int[:, 1] - pacf_x_int, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_pacf_int.add_trace(go.Scatter(x=lags, y=pacf_confint_int[:, 0] - pacf_x_int, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_pacf_int.update_layout(title=dict(text='PACF van originele en geïnterpoleerde tijdreeks'),
                          template="ggplot2")

# fig_pacf_int.show()

#%% Train-test split
# Test is 35 days
date_min = ts.index.min()
date_max = ts.index.max()
len_test = timedelta(days = 35)

# Split
ts_train = ts.loc[ts.index.max() - len_test - timedelta(days = 1):].copy()
ts_test = ts.loc[:ts.index.max() - len_test].copy()

fig_split = go.Figure()

fig_split.add_trace(go.Scatter(x=ts_train.index, y=ts_train['teruglevering'],
                               name = 'Train'))

fig_split.add_trace(go.Scatter(x=ts_test.index, y=ts_test['teruglevering'],
                               name = 'Train'))

fig_split.update_layout(title={'text':'Train-test split'})

# fig_split.show()

#%% Functies voor de modellen
ts = pd.read_csv('ts.csv', index_col = 'Meetdatum')
ts.index = pd.to_datetime(ts.index)
ts_model = pd.read_csv('ts_model.csv', index_col = 'index')
ts_model.index = pd.to_datetime(ts_model.index)
ts_train = pd.read_csv('ts_train.csv', index_col = 'index')
ts_train.index = pd.to_datetime(ts_train.index)
ts_test = pd.read_csv('ts_test.csv', index_col = 'index')
ts_test.index = pd.to_datetime(ts_test.index)
df_fourier = pd.read_csv('df_fourier.csv', index_col = 'index')
df_fourier.index = pd.to_datetime(df_fourier.index)

fourier_cols = ['sin365_1', 'cos365_1']

# Auto-ARIMA met FT
res_fourier = pm.auto_arima(ts_train['teruglevering_int_1_sc'], 
                        d = 1,
                        start_p = 0, max_p = 2,
                        start_q = 2, max_q = 3,
                        exogenous = ts_train[fourier_cols],
                        information_criterion = 'aic', trace = True, 
                        error_action = 'ignore', stepwise = True
                       )
print(res_fourier.summary())

def plot_fourier(df1, col1, name1, df2, col2, name2, title):
    """
    Returns plot met daarin fourier terms geplot
        
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1.index,
                                        y=df1[col1],
                                        name=col1,
                                        line_color='rgb(255,127,14)'
                                       ))
    
    fig.add_trace(go.Scatter(x=df2.index,
                                        y=df2[col2],
                                        name=col2,
                                        line_color='rgb(76,146,194)'
                                       ))

    fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right",
                                          x=1),
                                 title={
                                     'text': title,
                                     'y':0.86,
                                     'x': 0.08},
#                                  yaxis_title="Waarde (kWh)",
#                                  xaxis_title="Datum",
                                 template="ggplot2"
                                )

    return fig

def plot_fourier2(df1, col1, name1, df2, col2, name2, df3, col3, name3, title):
    """
    Returns plot met daarin fourier terms geplot
        
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1.index,
                                        y=df1[col1],
                                        name=name1,
                                        line_color='rgb(76,146,194)'
                                       ))
    
    fig.add_trace(go.Scatter(x=df2.index,
                                        y=df2[col2],
                                        name=name2,
                                        line_color='rgb(255,127,14)'
                                       ))
    
    fig.add_trace(go.Scatter(x=df3.index,
                                        y=df3[col3],
                                        name=name3,
                                        line_color='rgb(54,165,54)'
                                       ))    

    fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right",
                                          x=1),
                                 title={
                                     'text': title,
                                     'y':0.86,
                                     'x': 0.08},
#                                  yaxis_title="Waarde (kWh)",
#                                  xaxis_title="Datum",
                                 template="ggplot2"
                                )

    return fig


def plot_forecast(pred_col, title):
    """
    Plot de train, test en voorspelling op de train en testset
    
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts_test.index,
                                        y=ts_test['teruglevering'],
                                        name='Test',
                                        line_color='rgb(76,146,194)'
                                       ))
    fig.add_trace(go.Scatter(x=ts_train.index,
                                        y=ts_train['teruglevering'],
                                        name='Train',
                                        line_color='rgb(255,127,14)'
                                       ))
    fig.add_trace(go.Scatter(x=ts_test.index,
                                        y=ts_test[pred_col],
                                        name='Voorspelling test',
                                        line_color='rgb(0,0,0)'
                                       ))
    fig.add_trace(go.Scatter(x=ts_train.index,
                                        y=ts_train[pred_col],
                                        name='Voorspelling train',
                                        line_color='rgb(0,0,0)',
                                        line= dict(dash='dash')
                                       ))
    return fig

#%% Model 1 plotten
fig_FT_1 = plot_fourier(df_fourier, 'C1','C1', 
                        df_fourier, 'S1','S1',
                        'De Fourier terms die het jaarlijkse seizoenscomponent modelleren')
fig_FT_2 = plot_fourier(ts_model, 'teruglevering_int_1_sc', 'Gemodelleerd seizoenscomponent door Fourier term',
                        df_fourier, 'Fourier_1', 'Seizoen door Fourier terms',
                        'De Fourier terms die het jaarlijkse seizoenscomponent modelleren')
voorspelling_model1 = plot_forecast('pred_fourier1_rev', 'Voorspelling van model met Fourier term voor siezoenscomponent')

#%% Model 2 plotten
from statsmodels.tsa.statespace.sarimax import SARIMAX
res_zon1 = SARIMAX(ts_train['teruglevering_sc'], 
                   order = (4,1,0),
                   exog = ts_train[['zonneschijn_sc']],
                   trend = None
                  ).fit()
res_zon1.summary()
voorspelling_model2 = plot_forecast('pred_zon2_rev', 'Voorspelling van model met zonneschijn als extra regressor')

#%% Model 3 plotten
#acf en pacf van de wortel van de teruglevering
acf_x, acf_confint = gen_values_acf(ts['teruglevering_sqrt_sc'])
pacf_x, pacf_confint = gen_values_pacf(ts['teruglevering_sqrt_sc'])
lags = gen_lags(10)

fig_corr3 = make_subplots(rows=2, cols=1)

# Plot acf en pacf
plot_acf(acf_x, fig_corr3, 1, 1)
plot_pacf(pacf_x, fig_corr3, 2, 1)

#Conf int acf
fig_corr3.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1] - acf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
             row = 1, col=1)

fig_corr3.add_trace(go.Scatter(x=lags,y=acf_confint[:, 0] - acf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_corr3.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1] - pacf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr3.add_trace(go.Scatter(x=lags,y=pacf_confint[:, 0] - pacf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr3.update_layout(title={'text':'ACF en PACF voor de wortel van teruglevering'})

# acf en pacf gedifferentieerd
acf_x, acf_confint = gen_values_acf(ts['teruglevering_sqrt_sc'].diff().dropna())
pacf_x, pacf_confint = gen_values_pacf(ts['teruglevering_sqrt_sc'].diff().dropna())
lags = gen_lags(10)

fig_corr4 = make_subplots(rows=2, cols=1)

# Plot acf en pacf
plot_acf(acf_x, fig_corr4, 1, 1)
plot_pacf(pacf_x, fig_corr4, 2, 1)

#Conf int acf
fig_corr4.add_trace(go.Scatter(x=lags, y=acf_confint[:, 1] - acf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
             row = 1, col=1)

fig_corr4.add_trace(go.Scatter(x=lags,y=acf_confint[:, 0] - acf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 1, col=1)

#Conf int pacf
fig_corr4.add_trace(go.Scatter(x=lags, y=pacf_confint[:, 1] - pacf_x, # Bovengrens
                         fill=None,
                         mode='lines',
                         line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr4.add_trace(go.Scatter(x=lags,y=pacf_confint[:, 0] - pacf_x, # Ondergrens
                         fill='tonexty', fillcolor='rgba(0,204,204,0.3)', # fill area between trace0 and trace1
                         mode='lines', line_color='rgb(0,204,204)'),
              row = 2, col=1)

fig_corr4.update_layout(title={'text':'ACF en PACF voor de wortel van teruglevering gedifferentieerd'})

res_zon3 = pm.auto_arima(ts_train['teruglevering_sqrt_sc'], 
                        d = 1,
                        start_p = 0, max_p = 2,
                        start_q = 0, max_q = 4,
                        exogenous = ts_train[['zonneschijn_sc']],
                        information_criterion = 'aic', trace = True, 
                        error_action = 'ignore', stepwise = True
                       )
res_zon3.summary()

voorspelling_model3 = plot_forecast('pred_zon3_sqrd', 'Voorspelling van model op wortel van tergulevering met zonneschijn als extra regressor')

#%% Model 4 plotten
zon_fourier_cols = ['zonneschijn_sc',
                    'sin365_1', 'cos365_1',
                    'sin365_2', 'cos365_2',
                    'sin365_3', 'cos365_3']
res_zon_fourier = pm.auto_arima(ts_train['teruglevering_sqrt_sc'], 
                        d = 1,
                        start_p = 0, max_p = 2,
                        start_q = 2, max_q = 4,
                        exogenous = ts_train[zon_fourier_cols],
                        information_criterion = 'aic', trace = True, 
                        error_action = 'ignore', stepwise = True
                       )
res_zon_fourier.summary()

fig_FT_3 = plot_fourier2(df_fourier, 'Fourier_1_zf', 'Fourier_1_zf', 
                         df_fourier, 'Fourier_2_zf', 'Fourier_2_zf', 
                         df_fourier, 'Fourier_3_zf', 'Fourier_3_zf', 
                         'De Fourier terms die het jaarlijkse seizoenscomponent modelleren \nModel heeft ook zonneschijn als extra regressor')
fig_FT_4 = plot_fourier(ts, 'teruglevering_sqrt_sc', 'Wortel teruglevering',
                        df_fourier, 'Fourier_total_zf', 'Seizoen door Fourier terms op wortel van teruglevering',
                        'Gemodelleerd seizoenscomponent door Fourier terms')

voorspelling_model4 = plot_forecast('pred_zon_fourier_sqrd', 
              'Voorspelling van model met zonneschijn als extra regressor, Fourier term voor seizoenscomponent \nVoorspeld op wortel van teruglevering')

#%% Model 5 plotten
res_zon_temp = SARIMAX(ts_train['teruglevering_sc'], 
                       order = (4,1,0),
                       exog = ts_train[['zonneschijn_sc', 'temp_sc']],
                       trend = None
                      ).fit()

res_zon_temp.summary()

voorspelling_model5 = plot_forecast('pred_zon_temp_rev', 
              'Voorspelling van model met zonneschijn men temperatuur als extra regressors')

#%% Dashboard
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Tabs
app.layout = html.Div([
    html.H1('Voorspellen van zonne-energie'),
    dcc.Tabs(id="Tabs", value='Titelblad', children=[
        dcc.Tab(label='naampie', value='Titelblad'),
        dcc.Tab(label='Zonne-energie', value='Tab_tl'),
        dcc.Tab(label='Weer data', value='Tab_weer'),
        dcc.Tab(label='Data bewerkingen', value='Tab_bewerkingen'),
        dcc.Tab(label='Model Fourier Terms', value='Tab_FT'),
        dcc.Tab(label='Model Zonneschijnuren, zonder Fourier Terms', value='Tab_Z'),
        dcc.Tab(label='Model Zonneschijnuren en transformatie Teruglevering', value='Tab_Z2'),
        dcc.Tab(label='Model Zonneschijnuren, Fourier Terms en transformatie Teruglevering', value='Tab_Z_FT'),
        dcc.Tab(label='Model Zonneschijnuren en Temperatuur, zonder Fourier Terms', value='Tab_ZT'),
    ]),
    html.Div(id='Tabs_content')
])

@app.callback(Output('Tabs_content', 'children'),
              Input('Tabs', 'value'))
def render_content(tab):
    if tab == 'Titelblad':
        return html.Div([
            html.H3('Voorspellen van het energieverbruik van zonnepanelen'),
            html.Div([
                html.B('De opdracht:'),
                html.P('Steeds meer huishoudens stappen over naar zonnepanelen, dit komt doordat duurzame energie tegenwoordig enorm in is. Daarnaast is er de mogelijkheid om de energie die opgewekt wordt via de zonnepanelen terug te leveren aan het energienetwerk. (waarom is dat fijn?). Dit is erg fijn, maar hoe zit het met de de hoeveelheid energie die wordt opgewekt als de zon niet of nauwelijks schijnt. (bruggetje naar de opdracht), om hier een beter inzicht te krijgen is dan ook de vraag om een voorspellingsmodel te maken dat één week vooruit voorspelt wat het energieverbruik is door zonnepanelen.'),
                html.B('De werkwijze:'),
                html.P('De manier waarop te werk is gegaan is als volgt: Allereerst is er onderzoek gedaan naar de te verklaren variabele, de zonne-energie. Vervolgens is door middel van de overige variabelen te plotten en de onderlinge correlatie te onderzoeken besloten welke variabele er kunnen dienen als extra verklarend voor de zonne-energie. Nadat dat deze zijn vastgesteld is er databewerking gedaan, zo is er onderzocht of er outliers in de variabelen zitten en hoe de (partiële) autocorrelatie er uit ziet. Als laatste stap in de databewerking is er splitsing gemaakt in een train- en testdataset.'),
                html.P('Nadat alle bewerkingen voltooid zijn, zijn er verschillende modellen gemaakt. Er is begonnen met een model met alleen de zonne-energie als ... variabele. Vervolgens is stapje voor stapje uitgebreid en daarmee is het uiteindelijke model ontstaan.'),
                html.B('De datasets:'),
                html.P('De datasets die zijn gebruikt voor het onderzoek zijn: (precieze namen)'),
                html.B('De studenten:'),
                html.P('Lea van den Heuvel (10857020) en Ymke van der Waal (18071279)')
                ]),
            
            ])
    elif tab == 'Tab_tl':
        return html.Div([
            html.H3('Zonne-energie'),
            html.Div([
                html.P('Allereerst is er onderzoek gedaan naar de variabele die voorspeld gaat worden, de teruglevering van de zonne-energie. In de grafiek hieronder is de teruglevering van de zonne-energie in kWh te zien over de loop van de tijd. Daarnaast is ook de het gemiddelde over 7 dagen geplot.')
                ]),
                dcc.Graph(
                    id='Zonne_energie_rolling_mean_tl',
                    figure=fig_rolling_tl
                    ),
            html.H3('Histogram per seizoen'),
            html.Div([
                html.P('Vervolgens is er een histogram geplot van de teruglevering van de zonne-energie. In het histogram is er onderscheid gemaakt tussen de verschillende seizoenen. Wat opvalt is dat vooral in de lente en zomer veel energie wordt terugleverd ten opzichte van de herft en de winter.')
                ]),
                dcc.Graph(
                    id='Histogram_per_seizoen_tl',
                    figure=fig_hist2
                    ),
            html.H3('Boxplot per maand'),
            html.Div([
                html.P('Ook in de boxplot zijn er verschillen te zien in de teruglevering, voornamelijk in de zomermaanden wordt veel energie teruggeleverd. Er is een toename te zien vanaf de maand april en vanaf de maand september/oktober daalt de energieteruglevering weer. Wat vergelijkbaar is met het histogram hierboven.')
                ]),
                dcc.Graph(
                    id='Boxplot_per_maand',
                    figure=fig_boxplot
                    ),
            #Callback toevoegen!
            html.H3('Gemiddelde per dag(week/maand)/jaar'),
            html.Div([
                html.P('!!! Nog iets van tekst !!!')
                ]),
                dcc.Graph(
                    id='Gem_tl',
                    figure=fig_mean
                    ),
            html.H3('ACF en PACF'),
            html.Div([
                html.P('!!! Nog iets van tekst !!!')
                ]),
                dcc.Graph(
                    id='ACF_en_PACF_zonne_energie',
                    figure=fig_corr1
                    ),
            html.H3('ACF en PACF zonne-energie gedifferentiëerd'),
            html.Div([
                html.P('')
                ]),
                dcc.Graph(
                    id='ACF_en_PACF_zonne_energie_gedifferentiëerd',
                    figure=fig_corr2
                    ),
                ])
    elif tab == 'Tab_weer':
        return html.Div([
            # Callback toevoegen!
            html.H3('Weer data'),
            html.Div([
                html.P('Na het onderzoeken van de teruglevering is er gekeken naar de weersvariabelen. Onder de weersvariabelen worden verstaan: de duur van de zonneschijn, de temperatuur, de hoeveelheid neerslag en de duur van de neerslag. In de eerste grafiek zijn de weersvairabelen te zien over de loop der tijd. Daarnaast is ook het gemiddelde te zien over 7 dagen van de weersvariabelen.')
                ]),
                dcc.Graph(
                    id='Rolling_mean_weer',
                    figure=fig_rolling_zon
                    ),
            html.H3('Correlatie tussen zonne-energie en weer-data'),
            html.Div([
                html.P('Vervolgens is de correlatie onderzocht tussen de weervariabelen en de zonne-energie. Hieruit wordt geconcludeerd dat de duur van de zonneschijn en de temperatuur de hoogste correlatie hebben. Deze zijn dan ook gekozen om mee te nemen in het model.')
                ]),
                dcc.Graph(
                    id='Corr_weer_data',
                    figure=fig_corr
                    ),
            # Callback toevoegen!
            html.H3('Correlatie tussen zonne-energie en weer-data'),
            html.Div([
                html.P('Ten slotte wordt de lineaire regressie geplot tussen de teruglevering van de zonne-energie en de weersvariabelen.')
                ]),
                dcc.Graph(
                    id='Regplot_weer_tl',
                    figure=fig_reg
                    )
            ])
    elif tab == 'Tab_bewerkingen':
        return html.Div([
            html.H3('Resultaat Isolation Forest'),
            html.Div([
                html.P('Nadat alle variabelen zijn onderzocht is de data bewerkt zodat deze gebruikt kan worden voor de modellen. Het eerst wat gedaan wordt is het onderzoeken van de eventuele outliers. Hiervoor wordt het Isolation Forest alogritme gebruikt. (werking hierin of in notebook laten?)'),
                html.P('Uit de isolation forest van het eerste model volgen meerder outliers, er is voor gekozen om alleen de outliers aan de rechterkant te interpoleren. Vanwege het feit dat het model aan het begin van de tijdreeks nog aan het leren is en hierdoor meer datapunten als outliers ziet dan aan het eind van de tijdreeks. Met dit gegeven en de visualisatie van de outliers is besloten om de 5 meest rechter outliers te vervangen door middel van interpolatie.')
                ]),
                dcc.Graph(
                    id='IF_resultaat',
                    figure=fig_anom
                    ),
            html.H3('ACF'),
            html.Div([
                html.P('Vervolgens is voor zowel de orginele en bewerkte tijdreeks een acf en pacf plot gemaakt. Er is te zien dat er bij de ACF geen opmerkelijke veranderingen zijn.')
                ]),
                dcc.Graph(
                    id='ACF_int',
                    figure=fig_acf_int
                    ),
            html.H3('PACF'),
            html.Div([
                html.P('Bij de PACF wel; in de originele tijdreeks is lag 7 niet significant maar lag 8 wel, terwijl het in de bewerkte tijdreeks andersom is. Lag 6 is in de originele tijdreeks positief, maar in de bewerkte tijdreeks negatief.'),
                html.P('De verschillen zijn niet heel groot. Net zoals lag 7 in ded orignele tijdreeks ligt lag 8 in de bewerkte tijdreeks maar net buiten het betrouwbaarheidsinterval, en de correlatie van lag 6 is niet significant. Om deze reden is er dus voor gekozen te werken met de bewerkte tijdreeks voor het eerste model omdat hier geen extra regressors (zonneschijn of temperatuur) aan toegevoegd worden. Waar zonneschijn wordt toegevoegd als extra regressor wordt in eerste instantie gewerkt met de originele tijdreeks, omdat de correlatie tussen die twee erg hoor is.')
                ]),
                dcc.Graph(
                    id='PACF_int',
                    figure=fig_pacf_int
                    ),
            html.H3('Train-test split'),
            html.Div([
                html.P('Ten slotte wordt de data gesplitst in een train- en testdataset. De traindataset bestaat uit ... dagen/weken en de testdataset bestaat uit ... dagen/weken')
                ]),
                dcc.Graph(
                    id='Split',
                    figure=fig_split
                    )
                ])
    elif tab == 'Tab_FT':
        return html.Div([
            html.H3('Model met Fourier Terms om seizoen te modelleren'),
                dcc.Graph(
                    id='FT_1',
                    figure=fig_FT_1
                    ),
                dcc.Graph(
                    id='FT_2',
                    figure=fig_FT_2
                    ),
            html.H3('Voorspelling model met Fourier Terms'),
                dcc.Graph(
                    id='voorspelling_model1',
                    figure=voorspelling_model1
                    )
                ])
    elif tab == 'Tab_Z':
        return html.Div([
            html.H3('Model met Zonneschijnuren als voorspellende variabele zonder Fourier Terms modelleren'),
                dcc.Graph(
                    id='voorspelling_model2',
                    figure=voorspelling_model2
                    ),   
                ])
    elif tab == 'Tab_Z2':
        return html.Div([
            html.H3('Model met Zonneschijnuren en transformatie op de teruglevering modelleren'),
                dcc.Graph(
                    id='fig_corr3',
                    figure=fig_corr3
                    ),  
                dcc.Graph(
                    id='fig_corr4',
                    figure=fig_corr4
                    ), 
            html.H3('Voorspelling model Zonneschijnuren en transformatie van de teruglevering'),
                dcc.Graph(
                    id='voorspelling_model3',
                    figure=voorspelling_model3
                    ), 
                ])
    elif tab == 'Tab_Z_FT':
        return html.Div([
            html.H3('Model met Zonneschijnuren, Fourier Terms en tranformatie op de teruglevering modelleren'),
            html.H4('Resultaten seizoenscomponent met behulp van Fourier terms Model heeft ook zonneschijn als extra regressor'),
                dcc.Graph(
                    id='fig_FT_3',
                    figure=fig_FT_3
                    ),  
                dcc.Graph(
                    id='fig_FT_4',
                    figure=fig_FT_4
                    ),
            html.H3('Voorspelling Zonneschijnuren, Fourier Terms en tranformatie op de teruglevering'),
                dcc.Graph(
                    id='voorspelling_model4',
                    figure=voorspelling_model4
                    ), 
                ])
    elif tab == 'Tab_ZT':
        return html.Div([
            html.H3('Model met Zonneschijnuren en Temperatuur modelleren'),
                dcc.Graph(
                    id='voorspelling_model5',
                    figure=voorspelling_model5
                    ), 
                ])
    
if __name__ == '__main__':
    app.run_server(debug=True)







