import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")



class DataIngestion:
    def __init__(self): 
        """
        Initializes a new instance of the class.

        Parameters:
            None.

        Returns:
            None.
        """
        # Create a new instance of the DataIngestionConfig class
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.
        """
        
        # Log the start of the data ingestion step
        logging.info("Entered the data ingestion step")
        
        try:
            # Read the dataset as a dataframe
            df = pd.read_csv('notebook/Housing.csv')
            
            # Log that the dataset has been read
            logging.info('Read the dataset as dataframe')
            
            # Create directories for train and test data paths if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            # Save the dataframe as a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split started')

            # Split the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=8)

            # Save the train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train test split completed')

            # Return the train and test data paths
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
                
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)