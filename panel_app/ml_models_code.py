# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
import os
import panel as pn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import DateOffset
from datetime import datetime

# Import ML regression models
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# Import various regression metrics from sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score

# Define a list of metric functions and their corresponding names
metrics = [r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, max_error, explained_variance_score]
metric_names = ['r2_score', 'mean_absolute_error', 'mean_squared_error', 
                'median_absolute_error', 'max_error', 'explained_variance_score']

# Extend Panel functionality with the 'material' design template
pn.extension(design = 'material')


def split_scale_data(X, y, random_state_test = 1):
    """
    Splits the dataset into training and test sets, scales the features using StandardScaler, and returns these sets.

    Parameters
    ----------
    X : DataFrame
        DataFrame of the independent variables.
    y : DataFrame/Series
        Series of the dependent variable.
    random_state_test : int, optional
        Random state for train_test_split, by default 1.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test : DataFrame, DataFrame, Series, Series
        Returns the train and test sets for both independent and dependent variables.
    """
        
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state_test)

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)

    # Scale the training and test sets
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def time_split_scale_data(X, y, start='2005-4-1', end='2019-1-12'):
    """
    This function splits the input dataset based on a provided time range, then scales the features using the StandardScaler.

    Parameters:
    X: DataFrame - The input dataframe of independent variables.
    y: DataFrame/Series - The dependent variable series.
    start: string - The start date for splitting the data.
    end: string - The end date for splitting the data.

    Returns:
    The function returns the training and test datasets for the independent and dependent variables, with features scaled.
    """

    # Define train and test datasets based on the provided time range
    X_train_scaled = X.loc[start:end, :]
    X_test_scaled = X.loc[end:, :]
    y_train = y.loc[start:end]
    y_test = y.loc[end:]

    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit the StandardScaler to the features dataset
    X_scaler = scaler.fit(X)

    # Scale the features of the training and test datasets
    X_train_scaled = X_scaler.transform(X_train_scaled)
    X_test_scaled = X_scaler.transform(X_test_scaled)

    return X_train_scaled, X_test_scaled, y_train, y_test


    



def multi_target_linear_regression(df, target_list, start= '2005-4-1',  end= '2019-1-12'):
    """
    Performs linear regression for multiple targets in a DataFrame. Stores models, predictions, scores, and plots in a dictionary.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    target_list : list
        List of the target columns in the DataFrame.
    random_state_test : int, optional
        Random state for train_test_split, by default 1.

    Returns
    -------
    linear_reg_dict : dict
        A dictionary containing Linear Regression models, predictions, test data, scores, and plots for each target.
    """

    # Initialize an empty dictionary to store information for each target
    linear_reg_dict = {}

    # Create a DataFrame with metric names as indices
    metric_df = pd.DataFrame(index = metric_names)

    # Iterate through each target in the target list
    for a in target_list:
        # Initialize an empty dictionary for the current target
        linear_reg_dict[a] = {}

        # Initialize an empty list to store scores for each metric
        score_metrics = []

        # Define the target and feature set
        y = df[a]
        X = df.copy().drop(columns = a)

        # Split and scale the data
        X_train_scaled, X_test_scaled, y_train, y_test = time_split_scale_data(X, y, start, end)

        # Initialize and fit a Linear Regression model
        linear_reg_dict[a]['Linear_Reg_Model'] = LinearRegression()
        linear_reg_dict[a]['Linear_Reg_Model'].fit(X_train_scaled, y_train)

        # Generate predictions using the model
        linear_reg_dict[a]['Predict'] = pd.Series(linear_reg_dict[a]['Linear_Reg_Model'].predict(X_test_scaled), index = y_test.index, name = 'Prediction')

        # Store the test set labels
        linear_reg_dict[a]['y_Test'] = y_test

        # Generate a plot comparing the predictions to the test set values
        linear_reg_dict[a]['Plot_pred_test'] = linear_reg_dict[a]['Predict'].hvplot() * (linear_reg_dict[a]['y_Test']).hvplot()
        
        # Generate a plot comparing the predictions to the actual values
        linear_reg_dict[a]['Plot_pred_actual'] = linear_reg_dict[a]['Predict'].hvplot() * y.hvplot()
        
        # Compute scores for each metric and store them
        for b in metrics:
            score_metrics.append((b(linear_reg_dict[a]['y_Test'], linear_reg_dict[a]['Predict'])))
        
        # Create a DataFrame to store the scores
        score_df = pd.DataFrame(score_metrics, index = metric_df.index, columns = ['Metric'])
        # Store the score DataFrame in the dictionary for the current target
        linear_reg_dict[a]['Score'] = score_df

        # Compute and store feature importances
        linear_reg_dict[a]['Feature_Importance']= pd.DataFrame(linear_reg_dict[a]['Linear_Reg_Model'].coef_ , index = X.columns)

        # Generate and store a plot of feature importances
        linear_reg_dict[a]['Feature_Importance_Plot'] = linear_reg_dict[a]['Feature_Importance'].hvplot.bar(rot = 45, width = 800, height = 600)
        
    # After looping over all targets, return the dictionary containing all models, predictions, scores, and plots
    return linear_reg_dict




def multi_target__regression(df, target_list, start= '2005-4-1',  end= '2019-1-12', model = LGBMRegressor):
    """
    This function executes regression analysis for multiple target variables. For each target, it generates and stores a model,
    predictions, scores, and visualization plots in a dictionary.

    Parameters:
    df: DataFrame - The input dataframe.
    target_list: list - The list of target variables in the dataframe.
    start: string - The start date for splitting the data.
    end: string - The end date for splitting the data.
    model: ML model - The machine learning model to use for the regression (default is LGBMRegressor).

    Returns:
    A dictionary storing regression models, predictions, test data, scores, and plots for each target.
    """

    # Initialize an empty dictionary to store information for each target
    ml_reg_dict = {}

    # Create a DataFrame with metric names as indices
    metric_df = pd.DataFrame(index = metric_names)

    # Iterate through each target in the target list
    for a in target_list:
        # Initialize an empty dictionary for the current target
        ml_reg_dict[a] = {}

        # Initialize an empty list to store scores for each metric
        score_metrics = []

        # Define the target and feature set
        y = df[a]
        X = df.copy().drop(columns = a)

        # Split and scale the data
        X_train_scaled, X_test_scaled, y_train, y_test = time_split_scale_data(X, y, start, end)

        # Initialize and fit Regression model
        ml_reg_dict[a]['Reg_Model'] = model()
        ml_reg_dict[a]['Reg_Model'].fit(X_train_scaled, y_train)

        # Generate predictions using the model
        ml_reg_dict[a]['Predict'] = pd.Series(ml_reg_dict[a]['Reg_Model'].predict(X_test_scaled), index = y_test.index, name = 'Prediction')

        # Store the test set labels
        ml_reg_dict[a]['y_Test'] = y_test

        # Generate a plot comparing the predictions to the test set values
        ml_reg_dict[a]['Plot_pred_test'] = ml_reg_dict[a]['Predict'].hvplot() * (ml_reg_dict[a]['y_Test']).hvplot()
        
        # Generate a plot comparing the predictions to the actual values
        ml_reg_dict[a]['Plot_pred_actual'] = ml_reg_dict[a]['Predict'].hvplot() * y.hvplot()
        
        # Compute scores for each metric and store them
        for b in metrics:
            score_metrics.append((b(ml_reg_dict[a]['y_Test'], ml_reg_dict[a]['Predict'])))
        
        # Store the score DataFrame in the dictionary for the current target
        score_df = pd.DataFrame(score_metrics, index = metric_df.index, columns = ['Metric'])
        ml_reg_dict[a]['Score'] = score_df

        # Compute and store feature importances
        ml_reg_dict[a]['Feature_Importance']= pd.DataFrame(ml_reg_dict[a]['Linear_Reg_Model'].feature_importances_ , index = X.columns)

        # Generate and store a plot of feature importances
        ml_reg_dict[a]['Feature_Importance_Plot'] = ml_reg_dict[a]['Feature_Importance'].hvplot.bar(rot = 45, width = 800, height = 600)
        
    # After looping over all targets, return the dictionary containing all models, predictions, scores, and plots
    return ml_reg_dict
        

