from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_tuner import ModelTuner, ModelTunerConfig

class TrainPipeline:
    def __init__(self):
       
       try:
            obj=DataIngestion()
            raw_data_path = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, X_test, Y_train, Y_test, processor_path = data_transformation.initiate_data_transformation(raw_data_path)

            model_trainer = ModelTrainer()
            model_name = model_trainer.initiate_model_trainer(X_train, Y_train, X_test, Y_test)

            model_tuner = ModelTuner()
            model_tuner.initiate_model_tuner(model_name, X_train, Y_train, X_test, Y_test)

       except Exception as e:
            raise CustomException(e, sys)

        


        
if __name__=="__main__":
    logging.info("Starting the Training Pipeline")
    trainer = TrainPipeline()