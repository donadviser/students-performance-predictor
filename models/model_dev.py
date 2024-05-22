import sys

import numpy as np
import pandas as pd
import optuna

import xgboost as xgboost
#from lightgbm import LGBMRegressor
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

class ModelTrainer:
    """Model trainer class which trains the model
    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'lightgbm'.
        do_fine_tuning (bool, optional): Fine tuning. Defaults to True.

    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
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
                logger.info("Training Random Forest model completed successfully")
                return reg
            else:
                reg = RandomForestRegressor(
                    n_estimators=152,
                    max_depth=20,
                    min_samples_split=17,
                    random_state=42)
                reg.fit(self.X_train, self.y_train)
                logger.info("Training Random Forest model completed successfully")
                return reg                

        except Exception as e:
            logger.info("Error training Random Forest model")
            CustomException(e, sys)
            return None



        
