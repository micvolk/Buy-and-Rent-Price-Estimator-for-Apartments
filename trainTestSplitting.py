# -*- coding: utf-8 -*-
"""
Splits feature-engineered dataframes created by *featureEngineering.py* into
train and test dataframes for the models in module *modeling.py*.
Dependent variable y (rent-warm-price for rent-dataframe and buy-price for buy-dataframe)
and independent variables X are saved to separate files (this split is also performed for the whole dataframe).

OPTIONAL: Function add_nearestApartments_medianPrice_forModel() can be used
to get a price-estimate for every apartment based on its nearest-neighboors-apartments
as a new independent variable.
That can be interesting to use in linear-regression-models in module 'modeling.py',
as they are unable to handle Coordinates-information as independent-variables in a meaningful way.
But by default this function is not used because its very time-intensive to calculate.

@author: Michael Volk
"""

import generalFunctions as gf
from geopy import distance
from sklearn.model_selection import train_test_split


#----------------------------------------------------------------------------------------------------


# Section 1: Define functions for the modelling-process


def train_test_splitter(df, buy=True):
    """Returs training- and test-dataframe for explantory variables (X_train, X_test)
    and target-variables (y_train, y_test) on the basis of given df."""
        
    # Create y and X from df dependent if buy or rent dataframe is given
    if buy == True:
        y = df.Price_buy.copy()
        X = df.drop('Price_buy', axis=1).copy()
    else:
        y = df.Price_rent_cold.copy()
        X = df.drop('Price_rent_cold', axis=1).copy()
        
    # Split X and y in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return X, X_train, X_test, y, y_train, y_test

def add_nearestApartments_medianPrice_forModel(X_train, X_test=None, buy=True, n=10):    
    """
    Function can be used to get a price-estimate for every apartment based on its
    nearest-neighboors-apartments as a new independent variable.
    That can be interesting to use in linear-regression-models in module 'modeling.py',
    as they are unable to handle Coordinates-information as independent-variables in a meaningful way.
    CAUTION: By default this function is not used because its very time-intensive to calculate
    (Calculation time raises quadratic with number of rows!).
    
    Adds 'Nearest_price_perArea_forModel' and 'Price_estimate_nearest_forModel'
    which are calculated separately for the train- and test-data to avoid
    'train-test-contamination' (thats the difference to the function
    add_nearestApartments_medianPrice() in module 'featureEngineering.py').
    For the train-data (X_train) it is calculated on basis of the train-data.
    For the test-data (X_test) it is calculated only on basis of the train-data.
    
    'Nearest_price_perArea_forModel': Median Price per area of the n nearest apartments.
        for buy: Price_buy_per_Area
        for rent: Price_rent_cold_per_Area
    'Price_estimate_nearest_forModel': 'Nearest_price_perArea' multiplied with 'Area'
    Optional parameters:
        buy: True if X_train is dataframe for buy, False if X_train is dataframe for rent
        n: number of nearest neighbors to be included"""
    
    # Adds coordinates-tuple consisting of (Latitude, Longitude)"""
    X_train['Coordinates'] = list(zip(X_train.Latitude, X_train.Longitude))
    X_test['Coordinates'] = list(zip(X_test.Latitude, X_test.Longitude))
    
    def nearestApartments_medianPrice_perArea_X_train(row):
        """Returns the median Price per area of the n nearest apartments on the
        basis of X_train to given apartment row of dataframe X_train."""
        
        #Calculate the distance of each apartment of X_train to the given apartment row of X_train
        distances_to_row = X_train['Coordinates'].apply(lambda x: distance.distance(x, row['Coordinates']).km
                                                       if x.index != row['Coordinates'].index else 999999)
        # List the indices of the n nearest apartments
        indices = distances_to_row.sort_values().index[0:n]
        
        #Return the median price per area of the n nearest apartments
        if buy == True:
            return X_train.loc[indices].Price_buy_perArea.median()
        else:
            return X_train.loc[indices].Price_rent_cold_perArea.median()

    def nearestApartments_medianPrice_perArea_X_test(row):
        """Returns the median Price per area of the n nearest apartments on the
        basis of X_train to given apartment row of dataframe X_test."""
        
        #Calculate the distance of each apartment of X_train to the given apartment row of X_test
        distances_to_row = X_train['Coordinates'].apply(lambda x: distance.distance(x, row['Coordinates']).km
                                                       if x.index != row['Coordinates'].index else 999999)
        # List the indices of the n nearest apartments
        indices = distances_to_row.sort_values().index[0:n]
        
        #Return the median price per area of the n nearest apartments
        if buy == True:
            return X_train.loc[indices].Price_buy_perArea.median()
        else:
            return X_train.loc[indices].Price_rent_cold_perArea.median()
    
    #Calculate for all rows of X_train
    X_train['Nearest_price_perArea_forModel'] = X_train.apply(nearestApartments_medianPrice_perArea_X_train, axis=1)
    X_train['Price_estimate_nearest_forModel'] = X_train['Nearest_price_perArea_forModel'] * X_train['Area']

    #Calculate for all rows of X_test
    X_test['Nearest_price_perArea_forModel'] = X_test.apply(nearestApartments_medianPrice_perArea_X_test, axis=1)
    X_test['Price_estimate_nearest_forModel'] = X_test['Nearest_price_perArea_forModel'] * X_test['Area']
            
    return X_train, X_test


#----------------------------------------------------------------------------------------------------


# Section 2: Define the order of running the above functions.


def run():
    """
    Runs the above functions in defined order.
    """
    
    # Load dataframes
    df_buy = gf.load_data(filename = "featureEngineered_buy")
    df_rent = gf.load_data(filename = "featureEngineered_rent")
    
    # # Cut to only the first n rows of the dataframe (only relevant for testing)
    # n = 100
    # df_buy = df_buy[0:n].copy()
    # df_rent = df_rent[0:n].copy()
    
    # Execute train_test_splitter
    X_buy, X_train_buy, X_test_buy, y_buy, y_train_buy, y_test_buy = train_test_splitter(df_buy, buy=True)
    X_rent, X_train_rent, X_test_rent, y_rent, y_train_rent, y_test_rent = train_test_splitter(df_rent, buy=False)
    
    # # Execute add_nearestApartments_medianPrice_forModel
    # # CAUTION: Very time intensive to calculate. Calculation time raises quadratic with number of rows!
    # # Therefore this function is not used by default.
    # X_train_buy, X_test_buy = add_nearestApartments_medianPrice_forModel(X_train_buy, X_test_buy, buy=True, n=1)
    # X_train_rent, X_test_rent = add_nearestApartments_medianPrice_forModel(X_train_rent, X_test_rent, buy=False, n=1)
    
    # Save dataframes
    gf.save_data(X_buy, filename = "X_buy")
    gf.save_data(X_train_buy, filename = "X_train_buy")
    gf.save_data(X_test_buy, filename = "X_test_buy")
    gf.save_data(y_buy, filename = "y_buy")
    gf.save_data(y_train_buy, filename = "y_train_buy")
    gf.save_data(y_test_buy, filename = "y_test_buy")
    gf.save_data(X_rent, filename = "X_rent")
    gf.save_data(X_train_rent, filename = "X_train_rent")
    gf.save_data(X_test_rent, filename = "X_test_rent")
    gf.save_data(y_rent, filename = "y_rent")
    gf.save_data(y_train_rent, filename = "y_train_rent")
    gf.save_data(y_test_rent, filename = "y_test_rent")
 
    
# Function run() shall be executed if module is executed directly via console 
if __name__ == "__main__":
    run()