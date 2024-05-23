import sys

import numpy as np
import pandas as pd
import optuna

import xgboost as xgboost
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


from sklearn.base import RegressorMixin

from src.exceptions import CustomException
from src.logger import logger

class HyperparameterOptimisation:
    """Hyperparameter optimisation class which optimises the hyperparameters of the model
    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'lightgbm'.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test        
        self.y_test = y_test

    def optimise_randomforest(self, trial: optuna.Trial) -> float:
        """
        Optimises the hyperparameters of the model: Random Forest parameters

        Args:
            trial (optinua.Trial): Trial object

        Returns:
            Accurancy (str): metric like R-squared or mean squared error (depending on the problem type)
        """
         
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        
        return val_accuracy
    
    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing LightGBM.

        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        num_leaves = trial.suggest_int("num_leaves", 15, 50),
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        reg = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            #max_depth=max_depth,
            num_leaves=num_leaves,
        )
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        return val_accuracy
    
    def optimize_xgboost_regressor(self, trial: optuna.Trial) -> float:
        """
        Optimises the hyperparameters of the model: Xgboost parameters

        Args:
            trial (optinua.Trial): Trial object

        Returns:
            Accurancy (str): metric like R-squared or mean squared error (depending on the problem type)
        """
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-7, 10.0
            ),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = xgboost.XGBRegressor(**param)
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        return val_accuracy

class ModelTrainer:
    """Model trainer class which trains the model
    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'lightgbm'.
        do_fine_tuning (bool, optional): Fine tuning. Defaults to True.

    """

    def __init__(self, X_train: np.ndarray,  y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def random_forest_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """
        Fucntion that trains the RandomForest model

        Args:
            fine_tuning (bool, optional): Fine tuning. Defaults to True. If True, hyperparameter optimisation is performed
        """
        logger.info("Starting training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimisation(
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimise_randomforest, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                logger.info("Random Forest Best Parameters: {trial.params}")
                reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                    )
                reg.fit(self.X_train, self.y_train)
                print(f"Best Parameters:")
                #print(f"trial: {trial}")
                print(f"n_estimators: {n_estimators}")
                print(f"max_depth: {max_depth}")
                print(f"min_samples_split: {min_samples_split}")
                 
                logger.info("Training Random Forest model completed successfully")
                return reg
            else:
                reg = RandomForestRegressor(
                    n_estimators=152,
                    max_depth=20,
                    min_samples_split=17,
                    random_state=42)
                reg.fit(self.X_train, self.y_train)
                logger.info("Training Random Forest model with hyperparameter tunning completed successfully")
                return reg                

        except Exception as e:
            logger.info("Error training Random Forest model")
            CustomException(e, "Error in model_dev while training Random Forest")
            return None
        

    def lightgbm_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """
        It trains the LightGBM model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logger.info("Started training LightGBM model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimisation(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_lightgbm, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                #max_depth = trial.params["max_depth"]
                num_leaves = trial.params["num_leaves"]
                learning_rate = trial.params["learning_rate"]
                reg = LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    #max_depth=max_depth,
                    num_leaves=num_leaves,
                )
                reg.fit(self.X_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.X_train, self.y_train)
                logger.info("Training LightGBM model with hyperparameter tunning completed successfully")
                return model
        except Exception as e:
            logger.info("Error training LGBM model with hyperparameter tunning")
            CustomException(e, "Error in model_dev while training LGBM model")
            return None
            
        
    def xgboost_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """
        It trains the xgboost model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logger.info("Started training XGBoost model.")
        try:
            if fine_tuning:
                hy_opt = HyperparameterOptimisation(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost_regressor, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                reg = xgboost.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.X_train, self.y_train)
                logger.info("Training Xgboost with hyperparameter tunning completed successfully")
                return reg

            else:
                model = xgboost.XGBRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.X_train, self.y_train)
                logger.info("Training Xgboost completed successfully")
                return model
        except Exception as e:
            logger.info("Error training Xgboost")
            CustomException(e, "Error in model_dev while training Xgboost")
            return None




        
