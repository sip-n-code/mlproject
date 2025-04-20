import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model



# We need to create a config file for the model trainer (as we have done for data transformation and data ingestion components)
# and we need to create a model trainer class that will train the model and save the model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # all columns except the last one
                train_array[:, -1], # last column
                test_array[:, :-1], # all columns except the last one
                test_array[:, -1] # last column
            )

            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),  # Doesn't need random_state
                    "KNN": KNeighborsRegressor(),             # No random_state param
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor(verbose=False),
                    "AdaBoost": AdaBoostRegressor()
                }

            model_report: dict = evaluate_model(X_train=X_train, 
                                                y_train=y_train, 
                                                X_test= X_test, 
                                                y_test= y_test, 
                                                models=models)
            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)

            #To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print(best_model_name)

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")
            logging.info("Best found model on both training and test dataset")
            # Use this preprocessor object to transform the data
            # processing_obj = load_object(file_path=preprocessor_path)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, 
                        obj=best_model # Will be the pickle file of the model
                        )
            
            predicted= best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            