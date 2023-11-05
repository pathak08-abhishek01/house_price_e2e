import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    
    raw_data_path: str=os.path.join('artifacts',"data","data.csv")



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

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            
            # Save the dataframe as a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Return the train and test data paths
            return self.ingestion_config.raw_data_path
                
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

    


