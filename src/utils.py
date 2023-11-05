import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to a file.

    Parameters:
        file_path (str): The path to the file where the object will be saved.
        obj (Any): The object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.

    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    except Exception as e:
        raise CustomException(e, sys)
    
# creating the evaluation metrics
def evaluate_model(true, predicted):
    """
    Computes and returns the mean absolute error (MAE), root mean squared error (RMSE),
    and R^2 score for a given set of true and predicted values.

    Parameters:
        true (array-like): The true values.
        predicted (array-like): The predicted values.

    Returns:
        tuple: A tuple containing the MAE, RMSE, and R^2 score.
    """
    try:
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    """
    Load an object from a file.

    Args:
        file_path (str): The path to the file containing the object.

    Returns:
        The loaded object.

    Raises:
        CustomException: If there is an error loading the object from the file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)