# -*- coding: utf-8 -*-
"""
Loads all saved files created by executer.py for analysing them

@author: Michael Volk
"""

import generalFunctions as gf
import pickle


#----------------------------------------------------------------------------------------------------

  
# Section 1: Load the saved-files created by executer.py

  
# For buy:
scraped_buy = gf.load_data('scraped_buy')
cleaned_buy = gf.load_data('cleaned_buy')
featureEngineered_buy = gf.load_data('featureEngineered_buy')
X_buy = gf.load_data('X_buy')
X_train_buy = gf.load_data('X_train_buy')
X_test_buy = gf.load_data('X_test_buy')
y_buy = gf.load_data('y_buy')
y_train_buy = gf.load_data('y_train_buy')
y_test_buy = gf.load_data('y_test_buy')
errorMeasures_buy = gf.load_data('errorMeasures_buy')

# For rent:
scraped_rent = gf.load_data('scraped_rent')
cleaned_rent = gf.load_data('cleaned_rent')
featureEngineered_rent = gf.load_data('featureEngineered_rent')
X_rent = gf.load_data('X_rent')
X_train_rent = gf.load_data('X_train_rent')
X_test_rent = gf.load_data('X_test_rent')
y_rent = gf.load_data('y_rent')
y_train_rent = gf.load_data('y_train_rent')
y_test_rent = gf.load_data('y_test_rent')
errorMeasures_rent = gf.load_data('errorMeasures_rent')

#Load the nrwCityCoordinates.csv
nrwCityCoordinates = gf.load_data('nrwCityCoordinates')

# Load the saved modelConfigurations of the best model for buy and rent
modelConfigs = {}
for cat in ['_buy', '_rent']:
    with open(file= 'model' + cat + '.p', mode='rb') as pickled:
        data = pickle.load(pickled)
        modelConfigs['model' + cat] = data['model']
        modelConfigs['columns_used' + cat] = data['columns_used']
        modelConfigs['test_errors' + cat] = data['test_errors']
        

#----------------------------------------------------------------------------------------------------

  
# Section 2: Analysing and Debugging

