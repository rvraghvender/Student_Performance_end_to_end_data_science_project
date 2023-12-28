"""

"""

import os
import sys
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test  = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(), #
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(), #
                "Gradient Boosting" : GradientBoostingRegressor(), #
                "Random Forest Regressor" : RandomForestRegressor(), #
                "XGBRegressor" : XGBRegressor(), #
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), #
                "AdaBoost Regressor": AdaBoostRegressor() #
            }

            params={
                "K-Neighbors Regressor" :{
                    'n_neighbors': [3, 5, 7],            # Number of neighbors to consider
                    'weights': ['uniform', 'distance'],  # Weight function used in prediction
                    'p': [1, 2]                          # Power parameter for the Minkowski metric
                },

                "Lasso" :{
                        'alpha' : [0.001, 0.01, 0.1, 1.0, 10.0],
                        'random_state' : [42]
                },

                "Ridge" : {
                        'alpha' : [0.001, 0.01, 0.1, 1.0, 10.0],
                        'random_state' : [42]
                },

                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Linear Regression":{},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test,
                                                models=models, params=params)

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)

            logging.info("Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))