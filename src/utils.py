import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score



def save_object(file_path, obj):
    """
    Save the object to a file using pickle.
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)


def evaluate_model(models, X_train, y_train, X_test, y_test):
    """
    Evaluate the models and return a report of their performance.
    """
    model_report = {}
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square
            logging.info(f"{model_name} R2 Score: {r2_square}")
        return model_report
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)