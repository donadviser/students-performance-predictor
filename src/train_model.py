import sys
from typing import Annotated

import pandas as pd
from sklearn.base import RegressorMixin


from src.exceptions import CustomException
from src.logger import logger
from models.model_dev import ModelTrainer


def train_model(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame,
        model_type: str = 'randomforest', 
        do_fine_tuning: bool = True
        ) -> Annotated[RegressorMixin, "sklearn_regressor_model"]:
    """
    Trains the model

    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'randomforest'.
        do_fine_tuning (bool, optional): Fine tuning. Defaults to True.

    Returns:
        Annotated[RegressorMixin, "sklearn_regressor_model"]: Trained model
    """
    
    try:
        model_training  = ModelTrainer(X_train, y_train, X_test, y_test)
        if model_type == "randomforest":
            logger.info("Training random forest model")
            sklearn_regressor_model = model_training.random_forest_trainer(fine_tuning=do_fine_tuning)
            return sklearn_regressor_model
        else:
            raise ValueError('Model type not supported')

    except Exception as e:
        CustomException(e, sys)


if __name__ == '__main__':
    train_model()
