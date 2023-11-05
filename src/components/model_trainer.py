import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","models","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, X_train, Y_train, X_test, Y_test):
        

        try:
            logging.info("Splitting training and test input data")
            self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test

            models = {
            "LinearRegression" : LinearRegression(),
            "Lasso" : Lasso(),
            "Ridge" : Ridge(),
            "K-Neighbors Regressor" : KNeighborsRegressor(),
            "DecisionTreeRegressor" : DecisionTreeRegressor(),
            "RandomForestRegressor" : RandomForestRegressor(),
            "AdaBoostRegressor" : AdaBoostRegressor(),
            "XGBRegressor" : XGBRegressor(),
            "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
            }

            model_name_performance = {
                
            }

            for key, value in models.items():
                model = value
                model.fit(self.X_train, self.Y_train)
                Y_train_pred = model.predict(self.X_train)
                Y_test_pred = model.predict(self.X_test)
                # training performance
                mae_train, rmse_train, r2_square_train = evaluate_model(self.Y_train, Y_train_pred)

                logging.info(f"{model} Training Performance")
                logging.info(f"RMSE:{rmse_train}")
                logging.info(f"MAE:{ mae_train}")
                logging.info(f"R2 score {r2_square_train}")
                
                # testing performance
                mae_test, rmse_test, r2_square_test = evaluate_model(self.Y_test, Y_test_pred)

                logging.info(f"{model} Training Performance")
                logging.info(f"RMSE:{rmse_train}")
                logging.info(f"MAE:{ mae_train}")
                logging.info(f"R2 score {r2_square_train}")

                logging.info("#"*50)
                logging.info("\n")

                model_name_performance[key] = r2_square_test

            # sorting the dict with r2 score and get the name of the model with highest r2 score
            sorted_model_performance = sorted(model_name_performance.items(), key=lambda x: x[1], reverse=True)
            highest_r2_model = sorted_model_performance[0][0]

            if model_name_performance[highest_r2_model]<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            model_save_path = self.model_trainer_config.trained_model_file_path
          
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=highest_r2_model)
            
            logging.info(f"{highest_r2_model} saved")   
            return highest_r2_model
        
        except Exception as e:
            raise CustomException(e, sys)
                    
                

