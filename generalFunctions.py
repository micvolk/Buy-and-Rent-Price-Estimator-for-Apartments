# -*- coding: utf-8 -*-
"""
Contains own created functions which are necessary in more than one module
to avoid redundancy.

@author: Michael Volk
"""

import pandas as pd
from datetime import datetime

def dayTime():
    """Returns actual day and time formatted as String"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_data(filename):    
    """Read in data from csv-file and returns dataframe"""    
    return pd.read_csv(filename + '.csv')

def save_data(df, filename):
    """Saves given dataframe df to csv-file"""
    df.to_csv(filename + '.csv', index = False)
    print("Dataframe saved to file: " + filename + '.csv')
    return df

def get_numerical_columns(df):
    """Returns all numerical columns for given dataframe df as a list"""    
    return [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

def get_categorical_columns(df):
    """Returns all categorical columns for given dataframe df as a list"""    
    return [col for col in df.columns if df[col].dtype in ['object']]

def sort_df_columns(df):
    """Returns given dataframe df with columns sorted alphabetically"""
    columns = list(df.columns)
    columns.sort(key=lambda x: x.lower())
    return df[columns]

def columns_value_counts(df, columns, prefix=''):
    """Prints value_counts() for given dataframe df all regarding given columns with given prefix"""
    for col in columns:
        if prefix in col:
            print(df[col].value_counts() / df[col].value_counts().sum())
            print("\n------------------------------------------------------------------------------\n")