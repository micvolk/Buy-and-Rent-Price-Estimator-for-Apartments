# -*- coding: utf-8 -*-
"""
Imports all created modules and runs the full process
from scraping immowelt.de website, data-cleaning, feature-engineering,
model-building and saving the best model to be used for the app.

@author: Michael Volk
"""

import generalFunctions as gf
import scraping
import cleaning
import featureEngineering
import trainTestSplitting
import modeling


#----------------------------------------------------------------------------------------------------

  
# Section 1: Run the run() functions of the imported modules in defined order.
# Print day-timestamp after each step.

print('\n' + gf.dayTime() + ': EXECUTER STARTED!')
print('\n' + gf.dayTime() + ': scraping.py started')
scraping.run(maxNumberExposes=999999) #Optional: set parameter 'maxNumberExposes' for maximum number of exposes to be scraped
print('\n' + gf.dayTime() + ': cleaning.py started')
cleaning.run()
print('\n' + gf.dayTime() + ': featureEngineering.py started')
featureEngineering.run()
print('\n' + gf.dayTime() + ': trainTestSplitting.py started')
trainTestSplitting.run()
print('\n' + gf.dayTime() + ': modeling.py started')
modeling.run()
print('\n' + gf.dayTime() + ': EXECUTER FINISHED!')
