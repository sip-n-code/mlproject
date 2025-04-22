import os
import sys
from src.exception import CustomException
from src.logger import logging  
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # to create classes with default values
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.utils import save_object


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            df.columns = [col.replace(" ", "_").replace("/", "_") for col in df.columns]
            print(df.columns)
            logging.info("Dataset read as pandas dataframe")

            # Create the directory for artifacts if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into train and test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # python -m src.components.data_ingestion to run this file as a module
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_trainsformation = DataTransformation()
    train_arr, test_arr,_ = data_trainsformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

