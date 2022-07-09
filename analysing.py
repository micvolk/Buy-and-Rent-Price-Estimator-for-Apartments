# -*- coding: utf-8 -*-
"""
Loads all saved files created by executer.py for analysing them

@author: Michael Volk
"""

import generalFunctions as gf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        modelConfigs['test_errors_notAbsolute' + cat] = data['test_errors_notAbsolute']
        modelConfigs['test_errors_notAbsolute_Quantile_5%' + cat] = data['test_errors_notAbsolute'].quantile(q=0.05)
        modelConfigs['test_errors_notAbsolute_Quantile_95%' + cat] = data['test_errors_notAbsolute'].quantile(q=0.95)
        modelConfigs['test_errorfactor_notAbsolute_Quantile_5%' + cat] = np.exp(data['test_errors_notAbsolute'].quantile(q=0.05))
        modelConfigs['test_errorfactor_notAbsolute_Quantile_95%' + cat] = np.exp(data['test_errors_notAbsolute'].quantile(q=0.95))
        

#----------------------------------------------------------------------------------------------------

  
# Section 2: Analysing and Debugging
    
#Plotting distributions of different error measurement types
fig, ax = plt.subplots(3, 2, figsize=(10,10))
ax[0][0].set_title('Buy')
ax[0][1].set_title('Rent')
sns.histplot(modelConfigs['test_errors_buy'], stat='probability', shrink=0.8, ax=ax[0][0])
sns.histplot(modelConfigs['test_errors_rent'], stat='probability', shrink=0.8, ax=ax[0][1])
sns.histplot(modelConfigs['test_errors_notAbsolute_buy'], stat='probability', shrink=0.8, ax=ax[1][0])
sns.histplot(modelConfigs['test_errors_notAbsolute_rent'], stat='probability', shrink=0.8, ax=ax[1][1])
sns.histplot(np.exp(modelConfigs['test_errors_notAbsolute_buy']), stat='probability', shrink=0.8, ax=ax[2][0])
sns.histplot(np.exp(modelConfigs['test_errors_notAbsolute_rent']), stat='probability', shrink=0.8, ax=ax[2][1])
fig.tight_layout()