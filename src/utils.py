import os
import sys
import numpy as np
import pandas as pd
#import dill
import pickle
from src.exception import CustomException



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