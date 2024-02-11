import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# From every component we need to create a Config file or a class

@dataclass

class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts',"model.pkl")
class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                # Take all rows and all columns except the last one
                train_array[:,-1],
                # Take all rows and only the last column
                test_array[:,:-1],
                # Take all rows and all columns except the last one
                test_array[:,-1]
                # Take all rows and only the last column
            )
            models={
                "RandomForest":RandomForestRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "LinearRegression":LinearRegression(),
                "K-Neighbors Classifier":KNeighborsRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor()
            }
            
            model_report=dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            ## To get the best model from the model report
            
            best_model_score=max(sorted(model_report.values()))
            
            ## To get the best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)    
        
