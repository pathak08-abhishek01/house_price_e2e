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
        

def initiate_data_transformation(self, train_path, test_path):
    """
    This method initiates the data transformation.

    Args:
        train_path (str): The path to the training data file.
        test_path (str): The path to the testing data file.

    Returns:
        tuple: A tuple containing the transformed training and testing data, target feature data, and the path to the saved preprocessing object file.
    """
    try:
        # Read the train and test data from the specified paths
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Read train and test data completed")

        logging.info("Obtaining preprocessing object")

        # Get the data transformer object
        preprocessing_obj = self.get_data_transformer_object()

        target_column_name = "price"

        # Split the training data into input features and target features
        input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
        target_feature_train_df = train_df[target_column_name]

        # Split the testing data into input features and target features
        input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info(
            f"Applying preprocessing object on training dataframe and testing dataframe."
        )

        # Apply the preprocessing object on the training and testing input features
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        logging.info(f"Saved preprocessing object.")

        # Save the preprocessing object to a file
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
        )

        # Return the transformed data, target feature data, and the path to the saved preprocessing object file
        return (
            input_feature_train_arr, target_feature_train_df, input_feature_test_arr, target_feature_test_df,
            self.data_transformation_config.preprocessor_obj_file_path
        )
    





        
