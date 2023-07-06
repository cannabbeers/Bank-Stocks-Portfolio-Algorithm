# Import necessary modules and libraries
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import panel as pn
import os
# Import custom defined functions from the utility module
from utility import csv_to_dataframe, clean_timestamp, resample
# Import the StandardScaler module from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

# Set pandas options for maximum column and width display
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# Initialize the panel extension with 'material' design and 'dark' theme
pn.extension(design = 'material', theme = 'dark')

# Define paths to the different data files
stock_path = "Resources/Stock_Data/Daily/"
fred_daily = "Resources/FRED_Data/Daily/"
fred_weekly = "Resources/FRED_Data/Weekly/"
fred_monthly = "Resources/FRED_Data/Monthly/"
fred_quarterly ="Resources/FRED_Data/Quarterly/"

# Use the custom defined csv_to_dataframe function to convert the CSV files at the defined paths to dataframes
stocks_dfs = csv_to_dataframe(stock_path, True)
fred_daily_dfs = csv_to_dataframe(fred_daily, True)
fred_weekly_dfs = csv_to_dataframe(fred_weekly, True)
fred_monthly_dfs = csv_to_dataframe(fred_monthly, True)
fred_quarterly_dfs = csv_to_dataframe(fred_quarterly, True)

# Define list of stock column names and volume columns
stock_columns_list = [ /* specific column names */ ]
volume_columns = [ /* specific column names */ ]

# Define the columns to keep in the final dataframe for stock data and fred data
stock_data = [ /* specific column names */ ]
analysis_dropped_categories = [ /* specific column names */ ]
fred_data = [ /* specific column names */ ]

# Create a list of dataframes to concatenate for each time frequency
l_df = [stocks_dfs[a] for a in stocks_dfs.keys() if a not in ['Daily_SQ_df', 'Daily_SOFI_df', 'Daily_FRCB_df']]
l_daily = list(fred_daily_dfs.values())
l_weekly = list(fred_weekly_dfs.values())
l_monthly = list(fred_monthly_dfs.values())
l_quarterly = [fred_quarterly_dfs[a] for a in fred_quarterly_dfs.keys() if a != 'Quarterly_Net_perc_of_banks_tightening_standards_for_CRE_df']

# Merge the dataframes with the same frequency into one dataframe
daily_df = pd.concat(l_daily, axis = 1, join = 'inner')
weekly_df = clean_timestamp(pd.concat(l_weekly, axis = 1, join = 'inner'))
monthly_df = clean_timestamp(pd.concat(l_monthly, axis = 1, join = 'inner'))
quarterly_df = clean_timestamp(pd.concat(l_quarterly, axis = 1, join = 'outer'))

# Resample the different frequency dataframes to have the same timestamp as daily_df
quarterly_resampled_df = resample(quarterly_df, daily_df)
weekly_resampled_df = resample(weekly_df, daily_df)
monthly_resampled_df = resample(monthly_df, daily_df)

# Merge all the resampled dataframes into one dataframe
l_re = [weekly_resampled_df, daily_df, monthly_resampled_df, quarterly_resampled_df]
stocks_df = pd.concat(l_df, axis = 1, join = 'inner')
fred_df = pd.concat(l_re, axis=1, join ='inner')

# Concatenate the stocks
stocks_df = pd.concat(l_df, axis = 1, join = 'inner')
fred_df = pd.concat(l_re, axis=1, join ='inner')

# Concatenate the stocks dataframe and the fred dataframe into one single dataframe
stock_analysis_df = pd.concat([stocks_df, fred_df], axis =1, join='inner')

# Create a copy of the stock_analysis_df, drop unnecessary columns and replace any '.' with NaN values
stock_close_analysis_df = stock_analysis_df.copy().drop(columns = stock_columns_list)
stock_close_analysis_df = stock_close_analysis_df.drop(columns = analysis_dropped_categories)
stock_close_analysis_df = stock_close_analysis_df.replace( '.', np.nan)

# Resample stock_close_analysis_df to match its own frequency (this might be used to fill in any missing data)
stock_close_analysis_df = resample(stock_close_analysis_df,stock_close_analysis_df)

# Compute the correlation matrix of the final dataframe
corr_matrix = stock_close_analysis_df.corr()


