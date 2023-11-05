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
from src.utils import save_object, evaluate_model, load_object



@dataclass
class ModelTunerConfig:
    final_model_file_path: str = os.path.join("artifacts","models", "final_model.pkl")


class ModelTuner:
    def __init__(self):
        
        self.model_tuner_config = ModelTunerConfig()

    
    def initiate_model_tuner(self,model_name, X_train, Y_train, X_test, Y_test):
        try:
            self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test

            logging.info("Creating model object")
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
            
            

            self.model_name = model_name
            model = models[self.model_name]


            # Defining the hyperparameters
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                }
            

            # loadig the parameters
            logging.info("Loading the parameters")
            model_parameters = params[model_name]

            # Grid Search CV
            grid_search = GridSearchCV(model, model_parameters, cv=5, n_jobs=-1)
            grid_search.fit(self.X_train, self.Y_train)

            # print the training performance
            Y_train_pred = grid_search.predict(self.X_train)

            mae_train, rmse_train, r2_square_train = evaluate_model(self.Y_train, Y_train_pred)

            logging.info(f"{model} Training Performance")
            logging.info(f"RMSE:{rmse_train}")
            logging.info(f"MAE:{ mae_train}")
            logging.info(f"R2 score {r2_square_train}")

            # print the testing performance
            Y_test_pred = grid_search.predict(self.X_test)

            mae_test, rmse_test, r2_square_test = evaluate_model(self.Y_test, Y_test_pred)

            logging.info(f"{model} Training Performance")
            logging.info(f"RMSE:{rmse_train}")
            logging.info(f"MAE:{ mae_train}")
            logging.info(f"R2 score {r2_square_train}")



            # best parameters
            best_params = grid_search.best_params_
           

            # training the final model with the best params
            final_model = grid_search.best_estimator_
            final_model.fit(X_train, Y_train)

            # saving the model
            save_object(file_path=self.model_tuner_config.final_model_file_path, obj=final_model)

        except Exception as e:
            raise CustomException(e, sys)




            