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

pn.extension(design = 'material', theme = 'dark')


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


