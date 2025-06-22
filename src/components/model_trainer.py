import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
      AdaBoostRegressor,
      GradientBoostingRegressor,
      RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info("Splitting training and testing data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
            }

            model_report: dict = evaluate_model(models, X_train, y_train, X_test, y_test)
           

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 score > 0.6")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)
