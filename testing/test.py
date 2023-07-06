import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os
from ml_models_code import multi_target_linear_regression
from clean_data import stock_close_analysis_df, stock_analysis_df, stock_data, corr_matrix
import streamlit as st

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


#Script
stock_ml_dict = multi_target_linear_regression(stock_close_analysis_df, stock_data)

ml_features = list(stock_ml_dict['BAC_Close'].keys())


#Functions
def get_lin_reg_results(reg_dict, stock, result):
    return reg_dict[stock][result]

def corr_heat_plot(corr_matrix_df, target):
    fig = px.imshow(corr_matrix_df, x= corr_matrix_df[target])
    return fig

def corr_heat_plot2(corr_matrix_df, targets):
    fig = px.imshow(corr_matrix_df, x= corr_matrix_df[targets])
    return fig


#Widget Functions
def stock_select():
    return  st.selectbox("Stock", list(stock_ml_dict.keys()))

def results_widget():
    return st.selectbox("Linear Regression Results", ml_features )

def variable_widget():
    return st.selectbox("variable", list(stock_close_analysis_df.columns))

def stock_graph_multi_select():
    return st.multiselect('MultiSelect', list(stock_close_analysis_df.columns), default= ['BAC_Close'])

def variable_widget2():
    return st.multiselect("variable", list(corr_matrix.columns), default=['BAC_Close', 'C_Close'])


#Bound functions
def bound_plot():
    return get_lin_reg_results(stock_ml_dict, stock_select(),   results_widget())

def bound_bar_plot():
    return corr_heat_plot(corr_matrix, variable_widget())

def bound_bar_multi_plot2():
    return corr_heat_plot2(corr_matrix, variable_widget2())


#Apps
def ml_results_tab():
    return st.title('Bank Stock Portfolio Analysis'), st.header('ML Results'), st.write(bound_plot())
       
def stock_graphs():
    return st.header('Data Graphs'), st.plotly_chart(px.line(stock_analysis_df, y= stock_graph_multi_select(), width = 1000, height = 600))

def corr_graph1():
    return st.header('Stock Corr App1'), st.plotly_chart(bound_bar_plot())

def corr_graph2():
    return st.header('Stock Corr App2'), st.plotly_chart(bound_bar_multi_plot2())



#Full Application
def app_tabs_application():
    tab1, tab2, tab3, tab4 =st.tabs(["Stock Graphs", "Correlation", "Multi-Correlation", "Machine Learning Results"])

    with tab1:
        stock_graphs()

    with tab2:
        corr_graph1()

    with tab3:
        corr_graph2()

    with tab4:
        ml_results_tab()


app_tabs = app_tabs_application()