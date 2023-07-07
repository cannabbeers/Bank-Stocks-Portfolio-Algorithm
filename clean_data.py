import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from utility import csv_to_dataframe
from utility import clean_timestamp
from utility import resample
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# pn.extension(design = 'material', theme = 'dark')


stock_path = "Resources/Stock_Data/Daily/"
fred_daily = "Resources/FRED_Data/Daily/"
fred_weekly = "Resources/FRED_Data/Weekly/"
fred_monthly = "Resources/FRED_Data/Monthly/"
fred_quarterly ="Resources/FRED_Data/Quarterly/"


stocks_dfs = csv_to_dataframe(stock_path, True)
fred_daily_dfs = csv_to_dataframe(fred_daily, True)
fred_weekly_dfs = csv_to_dataframe(fred_weekly, True)
fred_monthly_dfs = csv_to_dataframe(fred_monthly, True)
fred_quarterly_dfs = csv_to_dataframe(fred_quarterly, True)


stock_columns_list = [
        'BAC_Open', 'BAC_High', 'BAC_Low', 'BAC_Adj Close',
        'C_Open', 'C_High', 'C_Low', 'C_Adj Close', 'FCNCA_Open', 'FCNCA_High', 'FCNCA_Low', 
        'FCNCA_Adj Close', 'FITB_Open', 'FITB_High', 'FITB_Low', 'FITB_Adj Close', 'JPM_Open', 
        'JPM_High', 'JPM_Low', 'JPM_Adj Close', 'SBNY_Open', 'SBNY_High', 'SBNY_Low', 'SBNY_Adj Close', 
        'SIVBQ_Open', 'SIVBQ_High', 'SIVBQ_Low', 'SIVBQ_Adj Close', 'USB_Open', 'USB_High', 'USB_Low', 'USB_Adj Close', 
        'VIX_Open', 'VIX_High', 'VIX_Low']

volume_columns = ["BAC_Volume", "C_Volume", "FCNCA_Volume", "FITB_Volume", 
                  "JPM_Volume", "SBNY_Volume", "SIVBQ_Volume", "USB_Volume"]


stock_data = ['BAC_Close', 'C_Close','FCNCA_Close',  
'FITB_Close',  'JPM_Close',  'SBNY_Close',  
'SIVBQ_Close',  'USB_Close',  "VIX_Close"]

#The following list are columns that have been dropped due to high correlation with other columns that are redundtant
analysis_dropped_categories = [
    "CRE_Loans_All_Commercial_Banks_CREACBW027SBOG",
    "loans_in_bank_credit_all_commercial_banks_TOTLL", "real_estate_loans_all_commercial_banks_RELACBW027SBOG", 
    "real_estate_loans_CRE_all_commercial_banks_CREACBW027SBOG", 'GDP_GDP','GDPC1_GDPC1', 'US_Recessions_by_GDP_indictators_JHDUSRGDPBR', 
    'Net_perc_of_large_bank_tightening_standards_for_credit_card_loans_SUBLPDCLCSLGNQ', 'Net_perc_banks_tightening_standards_commercial_industrial_loans_to_small_firms_DRTSCIS']


fred_data = ['bank_credit_all_commercial_banks_TOTBKCR', 
             'real_estate_loans_reisdential_revolving_home_equity_loans_RHEACBW027SBOG',
             'real_estate_loans_residential_real_estate_all_commercial_banks_RREACBW027SBOG',
             'Bank_Prime_Loan_Rate_DPRIME', 'Federal_Funds_Effective_Rate_DFF', 'CPI_Annual_Rate_CORESTICKM158SFRBATL', 
             'CPI_Percent_Change_CORESTICKM159SFRBATL', 'GDP_Based_Recession_Indicator_JHGDPBRINDX', 
             'Gov_Debt_to_GDP_GFDEGDQ188S', 'GPDC1_per_capita_A939RX0Q048SBEA', 'Household_Debt_to_GDP_HDTGPDUSQ163N', 
             'Interest_rates_and_price_indexes_CRE_BOGZ1FL075035503Q', 
             'Interest_Rates_Price_Indexes_Multi_Fanily_Estate_BOGZ1FL075035403Q', 
             'Interest_Rates_Price_Indexes_NYSE_BOGZ1FL073164003Q', 
             'Net_perc_dometic_banks_tightening_standards_for_commerical_and_industrial_loans_large_firms_DRTSCILM']


#remove below stocks because earlist date stock started trading was past 2010
#want to start 04/2005
l = list(stocks_dfs.keys())
l.remove('Daily_SQ_df')
l.remove('Daily_SOFI_df')
l.remove('Daily_FRCB_df')

l_df = list()
for a in l:
    l_df.append(stocks_dfs[a])


l_daily = list()
for a in fred_daily_dfs:
    l_daily.append(fred_daily_dfs[a])

l_weekly = list()
for a in fred_weekly_dfs:
    l_weekly.append(fred_weekly_dfs[a])

l_monthly = list()
for a in fred_monthly_dfs:
    l_monthly.append(fred_monthly_dfs[a])


l_keys = list(fred_quarterly_dfs.keys())

l_keys.remove('Quarterly_Net_perc_of_banks_tightening_standards_for_CRE_df')
l_quarterly = list()
for a in l_keys:
    l_quarterly.append(fred_quarterly_dfs[a])

daily_df = pd.concat(l_daily, axis = 1, join = 'inner')
weekly_df = pd.concat(l_weekly, axis = 1, join = 'inner')
weekly_df = clean_timestamp(weekly_df)
monthly_df = pd.concat(l_monthly, axis = 1, join = 'inner')
monthly_df = clean_timestamp(monthly_df)
quarterly_df = pd.concat(l_quarterly, axis = 1, join = 'outer')
quarterly_df = clean_timestamp(quarterly_df)


quarterly_resampled_df = resample(quarterly_df, daily_df)
weekly_resampled_df = resample(weekly_df, daily_df)
monthly_resampled_df = resample(monthly_df, daily_df)
l_re = [weekly_resampled_df, daily_df, monthly_resampled_df, quarterly_resampled_df]



stocks_df = pd.concat(l_df, axis = 1, join = 'inner')
fred_df = pd.concat(l_re, axis=1, join ='inner')

stock_analysis_df = pd.concat([stocks_df, fred_df], axis =1, join='inner')
stock_close_analysis_df = stock_analysis_df.copy().drop(columns = stock_columns_list)
stock_close_analysis_df = stock_close_analysis_df.drop(columns = analysis_dropped_categories)
stock_close_analysis_df = stock_close_analysis_df.replace( '.', np.nan)
stock_close_analysis_df = resample(stock_close_analysis_df,stock_close_analysis_df)


corr_matrix = stock_close_analysis_df.corr()

app_description = """

# Instructions for Using the Bank Stock Portfolio Analysis App

Welcome to our Banking Stock Portfolio Analysis application! This tool is designed to provide insights into banking stock data and help you make informed decisions. It's simple and intuitive to use. Please follow the instructions below to get started:

## How to Navigate

The application has a tabbed layout with three main sections: 

1. **Correlation**: This section provides a heatmap of the correlations between different variables in the stock data.
2. **Stock Graphs**: Here you can visualize the trend of various stocks over time.
3. **Machine Learning Results**: This tab displays results from our machine learning analysis.

To navigate between these sections, click on the tab headers at the top of the page.

## How to Interact

Within each tab, you can interact with the displayed data and visualizations using the Streamlit widgets provided. Here's what each widget allows you to do:

1. **Stock Dropdown**: This allows you to select the specific stock that you want to analyze or visualize. Click on the dropdown menu, and select the stock of your interest from the list.
   
2. **Linear Regression Results Dropdown**: This widget allows you to select the specific result of the linear regression analysis that you want to display.

3. **Variable Dropdown**: Use this to select a variable from the dataset for which you'd like to see its correlation heatmap.

4. **Multiselect Widgets**: These let you select multiple variables or stocks for your analysis. Click on the widget, and select the items of your interest from the dropdown list that appears.

## Viewing Graphs

The line plots and heatmaps update dynamically based on your selections from the dropdowns and multiselect widgets. You can hover over the plots to see specific values.

## Chat Support

For any questions or issues, you can use the chat widget located at the bottom right of the page. Click on the widget to open the chat box and type your message. Our support team will respond as soon as possible.

Happy exploring and we hope our tool assists you in your financial analysis and decision-making process!

"""


ml_results_description_dict = {
    'Linear_Reg_Model': """
## Linear_Reg_Model
This is a Linear Regression model that has been trained on the dataset for a particular stock. It has learned from the relationships in the training data and can now predict the target variable (the stock's closing price) based on other input features.
""",
    'Predict':"""
## Predict
After the model has been trained, it can make predictions on unseen data. These predictions are made on the test data - a subset of the entire dataset that the model has not seen during training. This allows us to evaluate how well the model might perform in real-world scenarios.
""",
    'y_Test':"""
## y_Test
These are the actual target values from the test data. We compare these true values with the model's predictions to assess the performance of the model.
""",
    'Plot_pred_test':"""
## Plot_pred_test'
This is a graphical representation of the model's predictions compared with the actual target values from the test data. The plot helps us visually see how well the model's predictions align with the real values.
""",
    'Plot_pred_actual':"""
## Plot_pred_actual
Similar to 'Plot_pred_test', this is a plot of the model's predictions against all actual target values, including both the training and test data. This plot helps to visualize how well the model performs across the entire dataset.
""",
    'Score':"""
## Score
This includes different performance metrics that quantify how well the model is doing. These metrics include the R^2 score (which measures the proportion of the variance for the dependent variable that's explained by the model), Mean Absolute Error (the average absolute difference between the predicted and actual values), and several others. The higher the R^2 and the lower the error measures, the better the model is performing.
""",
    'Feature_Importance':"""
## Feature_Importance
In a linear regression model, each feature (or input variable) is assigned a coefficient that represents its 'importance' or influence in predicting the target variable. A larger absolute value of the coefficient suggests that the feature has a stronger impact on the prediction. However, these importance values are dependent on the scale of the features, so care should be taken when interpreting them.

""",
    'Feature_Importance_Plot':"""
## Feature_Importance_Plot
This is a bar plot representing the importance of each feature. It allows us to see at a glance which features are most influential in the model's predictions. The features are represented on the x-axis and their coefficients (importance) on the y-axis.
"""
}

