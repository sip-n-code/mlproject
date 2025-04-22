import os
import sys
import numpy as np
import pandas as pd
#import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV
from src.logger import logging



def save_object(file_path, obj):    
    """
    Save the object to a file using pickle.
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate the model using different metrics.
    """
    import time

    time_start = time.time()
    
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param=params[list(models.keys())[i]] ######################

            logging.info(f"Beginning Grid Search for {model_name}")
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            logging.info(f"Grid Search completed for {model_name}")

            #model.fit(X_train, y_train)

            # Predicting the model for both train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        time_end = time.time()
        time_taken = time_end - time_start
        logging.info(f"Time taken for model evaluation: {time_taken} seconds")
        print(f"Time taken for model evaluation: {time_taken} seconds")
        logging.info(f"Model evaluation completed")        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load the object from a file using pickle.
    """
    
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)