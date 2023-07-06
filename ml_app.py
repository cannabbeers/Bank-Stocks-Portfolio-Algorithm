import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import panel as pn
from ml_models_code import multi_target_linear_regression
from clean_data import stock_close_analysis_df, stock_analysis_df, stock_data, corr_matrix


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


pn.extension(design = 'material', theme = 'default')  #, theme = 'dark'


#Script
stock_ml_dict = multi_target_linear_regression(stock_close_analysis_df, stock_data)

ml_features = list(stock_ml_dict['BAC_Close'].keys())

#Functions
def get_lin_reg_results(reg_dict, stock, result):
    return reg_dict[stock][result]

def corr_heat_plot(corr_matrix_df, target):
    return corr_matrix_df[target].hvplot.heatmap(width = 1000, height =600, cmap = 'bwr')

def corr_heat_plot2(corr_matrix_df, targets):
    return corr_matrix_df[targets].hvplot.heatmap(rot = 45, width = 1000, height =600, cmap = 'bwr')


#Widgets
stock_widget = pn.widgets.Select(name="Stock", options=list(stock_ml_dict.keys()))

results_widget = pn.widgets.Select(name="Linear Regression Results", options = ml_features )

variable_widget = pn.widgets.Select(name="variable", value='BAC_Close', options=list(stock_close_analysis_df.columns))

multi_select = pn.widgets.MultiSelect(name='MultiSelect', value= ['BAC_Close'],
    options= list(stock_close_analysis_df.columns), size = 4)

variable_widget2 = pn.widgets.MultiSelect(name="variable", value=['BAC_Close', 'C_Close'], options=list(corr_matrix.columns))

#Bound functions
bound_plot = pn.bind(get_lin_reg_results,reg_dict=stock_ml_dict, stock=stock_widget, result = results_widget)

bound_bar_plot = pn.bind(corr_heat_plot,corr_matrix_df=corr_matrix, target=variable_widget)

bound_bar_multi_plot2 = pn.bind(corr_heat_plot2, corr_matrix_df=corr_matrix, targets=variable_widget2)



#Apps
ml_results_app = pn.Column(stock_widget,results_widget, bound_plot)

stock_graph_app = pn.Column(multi_select, pn.bind(stock_analysis_df.hvplot, y= multi_select, width = 1000, height = 600))

stock_corr_app = pn.Column(variable_widget, bound_bar_plot)

stock_corr_app2 = pn.Column(variable_widget2, bound_bar_multi_plot2)

app_tabs = pn.Tabs(
    ('Stock Corr App1', stock_corr_app),
    ('Stock Corr App2', stock_corr_app2),
    ('Data Graphs App', stock_graph_app),
    ('ML Results App', ml_results_app),
    dynamic=True, sizing_mode='stretch_both'
).servable(
    title='Bank Stock Portfolio Analysis'
    )

pn.serve(app_tabs)
