# -*- coding: utf-8 -*-
"""
Building different prediction models on basis of 'trainTestSplitting.py',
than choosing the best one, and save it
to file for using as prediction-model in flask-app 'app.py' for end-user.
HINT: Path of web-application folder is defined for my machine and needs to be adjusted for other machines!

Thanks to Ken Jee for inspiration to this module:
https://github.com/PlayingNumbers/ds_salary_proj/blob/master/model_building.py#L42

@author: Michael Volk
"""

import generalFunctions as gf
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle


#----------------------------------------------------------------------------------------------------


# Section 1: Define functions for the modeling process

def gridSearch_fitAndPredict(X_train, y_train, X_test, y_test, columns, pipe,
                             param_grid={}, scoringMethod='neg_median_absolute_error', cvFolds=3):
    """Fits given 'pipe' to given dataframes 'X_train' and 'y_train' based on given 'columns' 
    via grid-search over given parameters 'param_grid'.
    Uses Cross-Validation-method with 'scoringMethod' and 'cvFolds'.
    Than predict 'y_test_predict' for given 'X_test' and calculate scoring-measures regarding the error to given 'y_test'.
    Returns dictionary with fitted pipe and scoring-measures."""
    
    gs = GridSearchCV(pipe, param_grid, scoring=scoringMethod, cv=cvFolds, n_jobs=-1)
    gs.fit(X_train[columns], y_train)
    crossValScore_mean = -gs.best_score_
    y_test_predict = gs.best_estimator_.predict(X_test[columns])
    test_errors = (y_test_predict - y_test).abs()
    test_mean_absolute_error = mean_absolute_error(y_test, y_test_predict)
    test_median_absolute_error = median_absolute_error(y_test, y_test_predict)
    test_explained_variance_score = explained_variance_score(y_test, y_test_predict)
    
    return {'pipe_with_model': gs.best_estimator_,
            'columns_used': columns,
            'crossValScore_mean_median_absolute_error': crossValScore_mean,
            'test_median_absolute_error': test_median_absolute_error,
            'test_mean_absolute_error': test_mean_absolute_error,
            'test_explained_variance_score': test_explained_variance_score,
            'y_test_predict': y_test_predict,
            'test_errors': test_errors}


#----------------------------------------------------------------------------------------------------


# Section 2: Define the order of running the above functions.


def run():
    """
    Runs the above functions in defined order.
    Building different prediction models for buy and rent, than choosing for each the best one,
    and save it to file for using as prediction-model in flask-app 'app.py'.
    """
    
    #Define standard set of columns for models
    columns_standard = ['Area',
                       'Rooms',
                       'ConstructionYear',
                       'Latitude',
                       'Longitude',
                       'EQ_CAT_unknown',
                       'EQ_CAT_floorApartment',
                       'EQ_CAT_apartment',
                       'EQ_CAT_maisonette',
                       'EQ_CAT_penthouse',
                       'EQ_CAT_terraceApartment',
                       'EQ_CAT_loft',
                       'EQ_CON_unknown',
                       'EQ_CON_firstOccupancy',
                       'EQ_CON_upscale',
                       'EQ_CON_maintained',
                       'EQ_CON_renovated',
                       'EQ_CON_needsRenovation',
                       'EQ_CON_refurbished',
                       'EQ_CON_needsRefurbishment',
                       'EQ_CON_partlyRenovated',
                       'EQ_OUT_balcony',
                       'EQ_OUT_garden',
                       'EQ_OUT_loggia',
                       'EQ_OUT_terrace']
    
    # Define set of columns to be log-transformed
    columns_logTransform = ['Area', 'Rooms']
    
    # Define experimental set of columns (only necessary for defined experimental models, which are uncommented by default)
    neighborsBasedPriceEstimate = ['Price_estimate_nearest_forModel']
    
    # Define dataframe-names to be loaded and used in further process
    df_names = ['buy', 'train_buy', 'test_buy', 'rent', 'train_rent', 'test_rent']
    
    # Define column-transformer for dataframe regarding logarithm
    def make_log_transformer(columns):
        """Returns log_transformer-object for given column-name-list"""
        log_transformer = ColumnTransformer(
            [('log', FunctionTransformer(np.log, np.exp), columns)],
            remainder='passthrough', verbose_feature_names_out=False)
        return log_transformer
    
    # Load dataframes in dictionaries
    X={}
    y={}
    for df_name in df_names:
        X[df_name] = gf.load_data(filename = 'X_' + df_name)
        y[df_name] = gf.load_data(filename = 'y_' + df_name)
        y[df_name] = y[df_name].squeeze()
            
    # Transform y to log(y) (better measure for equal-weighting relative erros between predicted and real value)
    for df_name in y:
        y[df_name] = np.log(y[df_name])
    
    # Define dictionaries to save model and measures after fit and test
    modelsAndMeasures = {}
    errorMeasures = {}
    
    # Loop for fitting and predicting different models for buy and rent dataframe
    for cat in ['_buy', '_rent']:
        
        # Random Forest model based on ['Latitude', 'Longitude', 'Area']
        modelsAndMeasures['rf_LaLoAr' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude', 'Area'],
            pipe = make_pipeline(RandomForestRegressor()),
            )

        # Random Forest model based on ['Latitude', 'Longitude', 'ConstructionYear']
        modelsAndMeasures['rf_LaLoYe' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude', 'ConstructionYear'],
            pipe = make_pipeline(RandomForestRegressor()),
            )

        # Random Forest model based on ['Area', 'ConstructionYear']
        modelsAndMeasures['rf_ArYe' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Area', 'ConstructionYear'],
            pipe = make_pipeline(RandomForestRegressor()),
            )
    
        # Random Forest model based on ['Latitude', 'Longitude', 'Area', 'ConstructionYear']
        modelsAndMeasures['rf_LaLoArYe' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude', 'Area', 'ConstructionYear'],
            pipe = make_pipeline(RandomForestRegressor()),
            )
    
        # Random Forest model based on columns_standard
        modelsAndMeasures['rf_columns_standard' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = columns_standard,
            pipe = make_pipeline(RandomForestRegressor()),
            )
    
        # Random Forest model with scaled X_log based on columns_standard
        modelsAndMeasures['rf_columns_standard_Xlog_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = columns_standard,
            pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in columns_standard]),
                                 StandardScaler(),
                                 RandomForestRegressor()),
            )
    
        # Grid-Search Optimisation of so far best Random Forest-Model (measured in cross-validation score):
        # Random Forest model based on columns_standard with parameters grid-search-optimised
        modelsAndMeasures['rf_columns_standard_gsOpt' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = columns_standard,
            pipe = make_pipeline(RandomForestRegressor()),
            param_grid = {
                "randomforestregressor__n_estimators": range(50, 250, 50),
                "randomforestregressor__criterion": ['squared_error', 'absolute_error'],
                "randomforestregressor__min_samples_leaf": [1, 2, 3]
            }
            )
        
        # # Experimental model including feature-engineered variable Price_estimate_nearest_forModel:
        # # Random Forest model with scaled X_log based on columns_standard  + Price_estimate_nearest_forModel
        # modelsAndMeasures['rf_columns_standard_and_neighborsBasedPriceEstimate_Xlog_scaled' + cat] = gridSearch_fitAndPredict(
        #     X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
        #     columns = columns_standard + neighborsBasedPriceEstimate,
        #     pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in columns_standard]),
        #                          StandardScaler(),
        #                          RandomForestRegressor()),
        #     )
        
        # K-Nearest Neighbors model with ['Latitude', 'Longitude']
        modelsAndMeasures['knn_LaLo' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude'],
            pipe = make_pipeline(KNeighborsRegressor())
            )
            
        # K-Nearest Neighbors model with scaled ['Latitude', 'Longitude']
        modelsAndMeasures['knn_LaLo_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude'],
            pipe = make_pipeline(StandardScaler(), KNeighborsRegressor()),
            )
                
        # K-Nearest Neighbors model with scaled ['Latitude', 'Longitude', 'Area', 'ConstructionYear']
        modelsAndMeasures['knn_LaLoArYe_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude', 'Area', 'ConstructionYear'],
            pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
            )

        # # Experimental model including feature-engineered variable Price_estimate_nearest_forModel:
        # # K-Nearest Neighbors model with scaled ['Latitude', 'Longitude', 'Area', 'ConstructionYear', 'Price_estimate_nearest_forModel']
        # modelsAndMeasures['knn_LaLoArYe_and_neighborsBasedPriceEstimate_scaled' + cat] = gridSearch_fitAndPredict(
        #     X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
        #     columns = ['Latitude', 'Longitude', 'Area', 'ConstructionYear', 'Price_estimate_nearest_forModel'],
        #     pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
        #     )
        
        # K-Nearest Neighbors model with scaled columns_standard
        modelsAndMeasures['knn_columns_standard_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = columns_standard,
            pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
            )
        
        # Linear regression model based on 'Area'
        modelsAndMeasures['lr_Area' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Area'],
            pipe = make_pipeline(LinearRegression())   
            )
        
        # Linear regression model with log('Area')
        modelsAndMeasures['lr_Area_log' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Area'],
            pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in ('Area')]), LinearRegression())
            )
        
        # Linear regression model with scaled log('Area')
        modelsAndMeasures['lr_Area_log_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Area'],
            pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in ('Area')]),
                                 StandardScaler(),
                                 LinearRegression())
            )
        
        # Linear regression model with scaled ['Latitude', 'Longitude', 'Area', 'ConstructionYear']
        modelsAndMeasures['lr_LaLoArYe_Xlog_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = ['Latitude', 'Longitude', 'Area', 'ConstructionYear'],
            pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in ['Latitude', 'Longitude', 'Area', 'ConstructionYear']]),
                                 StandardScaler(),
                                 LinearRegression())
            )

        
        # Linear regression model with scaled X_log based on columns_standard
        modelsAndMeasures['lr_columns_standard_Xlog_scaled' + cat] = gridSearch_fitAndPredict(
            X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
            columns = columns_standard,
            pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in columns_standard]),
                                 StandardScaler(),
                                 LinearRegression())
            )
    
        # # Experimental model including feature-engineered variable Price_estimate_nearest_forModel:
        # # Linear regression model with scaled X_log based on columns_standard + Price_estimate_nearest_forModel
        # modelsAndMeasures['lr_columns_standard_and_neighborsBasedPriceEstimate_Xlog_scaled' + cat] = gridSearch_fitAndPredict(
        #     X['train' + cat], y['train' + cat], X['test' + cat], y['test' + cat],
        #     columns = columns_standard + neighborsBasedPriceEstimate,
        #     pipe = make_pipeline(make_log_transformer([col for col in columns_logTransform if col in columns_standard]),
        #                          StandardScaler(),
        #                          LinearRegression())
        #     )    
    
    
        # Construct dataframe with error-measures of fitted models for comparision
        # Errors are retransformed with exp() to get Error-Factors (=max(predicted, real)/min(predicted, real))
        errorMeasures['raw' + cat] = [[key,
                                      np.exp(modelsAndMeasures[key]['crossValScore_mean_median_absolute_error']),
                                      np.exp(modelsAndMeasures[key]['test_median_absolute_error']),
                                      np.exp(modelsAndMeasures[key]['test_mean_absolute_error']),
                                      modelsAndMeasures[key]['test_explained_variance_score'],
                                      np.exp(modelsAndMeasures[key]['test_errors'].quantile(q=0.50)),
                                      np.exp(modelsAndMeasures[key]['test_errors'].quantile(q=0.75)),
                                      np.exp(modelsAndMeasures[key]['test_errors'].quantile(q=0.90)),
                                      np.exp(modelsAndMeasures[key]['test_errors'].quantile(q=0.95)),
                                      ] for key in modelsAndMeasures if cat in key]
        errorMeasures['final' + cat] = pd.DataFrame(
                                                    np.array(errorMeasures['raw' + cat]),
                                                    columns=['Model',
                                                            'crossValScore_mean_median_absolute_error',
                                                            'test_median_absolute_error',
                                                            'test_mean_absolute_error',
                                                            'test_explained_variance_score',
                                                            'test_errorfactors_quantile_50%',
                                                            'test_errorfactors_quantile_75%',
                                                            'test_errorfactors_quantile_90%',
                                                            'test_errorfactors_quantile_95%'])
        errorMeasures['final' + cat] = errorMeasures['final' + cat].astype(
            {col: 'float64' for col in errorMeasures['final' + cat].columns if col != 'Model'})
        # Save errorMeasures['final' + cat] to csv-file
        gf.save_data(errorMeasures['final' + cat], filename = "errorMeasures" + cat)
    
        # Save model (=pickle the model) with the best 'crossValScore_mean_median_absolute_error'
        # Get name of model with the best 'crossValScore_mean_median_absolute_error'
        modelName = errorMeasures['final' + cat].loc[
            errorMeasures['final' + cat]['crossValScore_mean_median_absolute_error'].idxmin(),
            'Model']
        
        # Get model with used columns and test_errors and save it to file in current folder and web-application folder
        # HINT: Path of web-application folder is defined for my machine and needs to be adjusted for other machines!
        for path in ['', '../buy-rent-price-estimator-for-apartments_web-application/']:
            pickle.dump({'model': modelsAndMeasures[modelName]['pipe_with_model'],
                         'columns_used': modelsAndMeasures[modelName]['columns_used'],
                         'test_errors': modelsAndMeasures[modelName]['test_errors']},
                         open(path + 'model' + cat + '.p', 'wb'))
            print('Best model for ' + cat[1:] + ': ' + modelName + ' - saved to file ' + path + 'model' + cat + '.p')
        
     
# Function run() shall be executed if module is executed directly via console 
if __name__ == "__main__":
    run()