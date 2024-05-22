import sys
import os
from typing import Annotated, Tuple

import pandas as pd
from models.data_cleaning import DataCleaning
from src.exceptions import CustomException
from src.logger import logger

def clean_data(data: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.DataFrame, "y_train"],
        Annotated[pd.DataFrame, "y_test"],
    ]:
    """Data cleaning fuction which prepropocess the data and divides it into train and test data
    Args:
        data (pd.DataFrame): Data to be cleaned
        
    """
    try:
        logger.info("Cleaning data")
        data_cleaning = DataCleaning(data)
        df = data_cleaning.preprocess_data()
        X_train, X_test, y_train, y_test = data_cleaning.divide_data(df)
        logger.info("Cleaning data completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        CustomException(e, sys)