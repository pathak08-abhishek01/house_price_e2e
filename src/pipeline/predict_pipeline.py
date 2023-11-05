import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        """
        Predicts the output for a given set of features.

        Parameters:
            features (list): A list of features to be used for prediction.

        Returns:
            pred: The predicted output based on the given features.

        Raises:
            CustomException: If there is an error during the prediction process.
        """
        try:
            preprocessor_path = 'artifacts/transformer/preprocessor.pkl'
            model_path = 'artifacts/models/final_model.pkl'

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)
        
        

class CustomData:
    def __init__(self,area: float ,bedrooms: int,bathrooms: int,stories: int,parking: int,mainroad: str,
                 guestroom: str ,basement: str,hotwaterheating: str,airconditioning: str,
                 furnishingstatus: str,prefarea: str):
        """
        Initialize the object with the given parameters.

        Parameters:
            area (float): The area of the property.
            bedrooms (int): The number of bedrooms in the property.
            bathrooms (int): The number of bathrooms in the property.
            stories (int): The number of stories in the property.
            parking (int): The number of parking spaces in the property.
            mainroad (str): Indicates if the property is located near the main road.
            guestroom (str): Indicates if the property has a guest room.
            basement (str): Indicates if the property has a basement.
            hotwaterheating (str): Indicates if the property has hot water heating.
            airconditioning (str): Indicates if the property has air conditioning.
            furnishingstatus (str): The furnishing status of the property.
            prefarea (str): The preferred area of the property.
        """
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.stories = stories
        self.parking = parking
        self.mainroad = mainroad
        self.guestroom = guestroom
        self.basement = basement
        self.hotwaterheating = hotwaterheating
        self.airconditioning = airconditioning
        self.furnishingstatus = furnishingstatus
        self.prefarea = prefarea 


    def get_data_as_data_frame(self):
        """
        Generates a pandas DataFrame object with the given data.

        Returns:
            DataFrame: The generated DataFrame object.

        Raises:
            CustomException: If an error occurs while generating the DataFrame.
        """
        try:
            custom_data_input_dict = {
                "area": [self.area],
                "bedrooms": [self.bedrooms],
                "bathrooms": [self.bathrooms],
                "stories": [self.stories],
                "parking": [self.parking],
                "mainroad": [self.mainroad],
                "guestroom": [self.guestroom],
                "basement": [self.basement],
                "hotwaterheating": [self.hotwaterheating],
                "airconditioning": [self.airconditioning],
                "furnishingstatus": [self.furnishingstatus],
                "prefarea": [self.prefarea],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        