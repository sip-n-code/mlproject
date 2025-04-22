import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_data_path: str = os.path.join('artifacts', 'train_transformed.csv')
    transformed_test_data_path: str = os.path.join('artifacts', 'test_transformed.csv')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = None

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline.
        It handles both numerical and categorical features.
        It uses StandardScaler for numerical features and OneHotEncoder for categorical features.
        """
        
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]  


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # handle with median because we have outliers
                ('scaler', StandardScaler())
            ])

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # handle with most frequent/mode
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),  # handle unknown categories
                ('scaler', StandardScaler(with_mean=False))  # StandardScaler does not support sparse matrices, so we set with_mean=False
            ])

            logging.info("Categorical coloumns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, numerical_features), # pipeline name, pipeline, columns
                    ('categorical_pipeline', cat_pipeline, categorical_features) # pipeline name, pipeline, columns
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation process.
        It reads the train and test data, applies the transformation pipeline, and saves the transformed data.
        """
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Separate features and target variable from training and testing data
            input_feature_train_df = train_df.drop(columns=['math_score'])
            target_feature_train_df = train_df['math_score']

            input_feature_test_df = test_df.drop(columns=['math_score'])
            target_feature_test_df = test_df['math_score']

            logging.info("Separated features and target variable completed")

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            # Fit the preprocessor on the training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            logging.info("Fitting of preprocessor completed")

            logging.info("Saving preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)

            