import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        logging.info('Data Transformation has started')
        
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                
            ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    # Simple Imputer is handling the missing values
                    ('std_scaler',StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    # Most_frequent is the mode. It is handling the missing values
                    ('onehot',OneHotEncoder()),
                    ('std_scaler',StandardScaler())
                ]
            )
            
            logging.info("Numerical columns: {numerical_columns}")
            logging.info('Categorical columns: {categorical_columns}')
            
            
            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,numerical_columns),
                 ('cat_pipeline',cat_pipeline,categorical_columns)]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
     
    def initiate_data_transformation(self,train_path,test_path):
         
         try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)  
            
            logging.info('Read train and test data')
            
            logging.info('Obtaining preprocessing object')
            preprocessor=self.get_data_transformer_obj()
            
            target_column='math_score'
            numerical_columns=['writing_score','reading_score']
            
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]  
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column] 
            
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df=preprocessor.transform(input_feature_test_df) 
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)
                            ]
            
            test_arr=np.c_[input_feature_test_df,np.array(target_feature_test_df)]  
            logging.info(f'Saved the preprocessing object ')
            
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            ) 
         except Exception as e:
             raise CustomException(e,sys)       
