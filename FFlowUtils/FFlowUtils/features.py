from storeSalesUtils.prophetPipeline import median_filter, RMSLE

import pandas as pd
import numpy as np
import os

import warnings

def RMSLE(y, yhat):
    "Compute Root Mean Squared Log Error"
    metric = (np.log(1+yhat) - np.log(1+y))**2
    return round(np.sqrt(metric.mean()), 3)

def median_filter(df, varname = None, window=24, std=3): 
    """
    A simple median filter, removes (i.e. replace by np.nan) observations that exceed N (default = 3) 
    tandard deviation from the median over window of length P (default = 24) centered around 
    each observation.
    Parameters
    ----------
    df : pandas.DataFrame
        The pandas.DataFrame containing the column to filter.
    varname : string
        Column to filter in the pandas.DataFrame. No default. 
    window : integer 
        Size of the window around each observation for the calculation 
        of the median and std. Default is 24 (time-steps).
    std : integer 
        Threshold for the number of std around the median to replace 
        by `np.nan`. Default is 3 (greater / less or equal).
    Returns
    -------
    dfc : pandas.Dataframe
        A copy of the pandas.DataFrame `df` with the new, filtered column `varname`
    """
    
    dfc = df.loc[:,[varname]]
    
    dfc['median']= dfc[varname].rolling(window, center=True).median()
    
    dfc['std'] = dfc[varname].rolling(window, center=True).std()
    
    dfc.loc[dfc.loc[:,varname] >= dfc['median']+std*dfc['std'], varname] = np.nan
    
    dfc.loc[dfc.loc[:,varname] <= dfc['median']-std*dfc['std'], varname] = np.nan
    
    return dfc.loc[:, varname]

def make_shifted(df, shift_col, minlag, maxlag):
    """
    It takes a dataframe, a column name, and a range of lags, and returns a new dataframe with the
    original column and the lagged columns
    
    Args:
      df: the dataframe you want to shift
      shift_col: the column to shift
      minlag: the minimum lag to use
      maxlag: the maximum number of days to shift the data back
    
    Returns:
      A new dataframe with the shifted columns
    """
    new_df = df.copy()
    for i in range(minlag, maxlag+1):
        new_df.loc[:, '%s_%i'%(shift_col, i)] = new_df[shift_col].shift(i)
        
    return new_df

def backshift_df(df, shift_vars, minlag, maxlag):
    """
    It takes a dataframe, a list of variables to shift, and a minimum and maximum lag, and returns a
    dataframe with the shifted variables.
    
    Args:
      df: the dataframe you want to shift
      shift_vars: the variables to shift
      minlag: the minimum lag to create
      maxlag: the maximum number of lags to create
    
    Returns:
      A dataframe with the original data and the shifted data.
    """
    
    for var in shift_vars:
        df = make_shifted(df, var, minlag, maxlag)
    
    return df

def add_date_features(df, date_col):
    """
    It takes a dataframe and a date column as input, and returns a dataframe with the following date
    features: year, month, day, dayofyear, dayofweek, weekofyear, is_weekend
    
    Args:
      df: the dataframe
      date_col: The name of the column that contains the date
    
    Returns:
      A dataframe with the date features added.
    """
    # Date Features
    df_dateFeatures = df.copy()
    df_dateFeatures['year'] = df_dateFeatures[date_col].dt.year
    df_dateFeatures['month'] = df_dateFeatures[date_col].dt.month
    df_dateFeatures['day_number'] = df_dateFeatures[date_col].dt.day
    df_dateFeatures['dayofyear'] = df_dateFeatures[date_col].dt.dayofyear
    df_dateFeatures['dayofweek'] = df_dateFeatures[date_col].dt.dayofweek
    df_dateFeatures['weekofyear'] = df_dateFeatures[date_col].dt.weekofyear
    df_dateFeatures['is_weekend'] = np.where(df_dateFeatures.dayofweek.isin([5, 6]), 1, 0)
    
    return df_dateFeatures