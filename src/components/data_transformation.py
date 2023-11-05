import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    """
    This class contains the configuration for the data transformation.
    """
    preprocessor_obj_file_path = os.path.join("artifacts","transformer","preprocessor.pkl")
    
    



class DataTransformation:
    """
    This class contains the methods for the data transformation.
    """
    def __init__(self):
        """
        Initializes the DataTransformation object.
        """
        # Create a new instance of DataTransformationConfig
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method returns the data transformer object.
        """

        try:
            # Define the list of numerical columns
            numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            
            # Define the list of categorical columns
            categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']

            # Create a pipeline for numerical columns
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Create a pipeline for categorical columns
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder()),
            ])

            # Log the categorical and numerical columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipelines", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
            

    def initiate_data_transformation(self, raw_data_path):
        """
        This method initiates the data transformation.

        Args:
            train_path (str): The path to the training data file.
            test_path (str): The path to the testing data file.

        Returns:
            tuple: A tuple containing the transformed training and testing data, target feature data, and the path to the saved preprocessing object file.
        """
        try:
            self.data_path = raw_data_path
            # Read the train and test data from the specified paths
            data = pd.read_csv(self.data_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the data transformer object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"

            # Split the data into input features and target features
            X = data.drop(columns=[target_column_name],axis=1)
            Y= data[target_column_name]


            # Split the data into train and test sets
            logging.info("Splitting train and test data")
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the preprocessing object on the training and testing input features
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
            )
            
            # Return the transformed data, target feature data, and the path to the saved preprocessing object file
            return X_train, X_test, Y_train, Y_test, self.data_transformation_config.preprocessor_obj_file_path
                
            

        except Exception as e:
            raise CustomException(e, sys)
    





        
