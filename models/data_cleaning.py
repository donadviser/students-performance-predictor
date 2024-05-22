import sys
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logger

class DataCleaning:
    """Data cleaning class which preproceses the data and dvivides it into train and test datasets
    Args:
        data (pd.DataFrame): Data to be cleaned
        
    """

    def __init__(self, ) -> None:
        self.df = data

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data: Removes columns which are not required, fills in missing values, and converts the data types"""

        try:
            #self.df = self.df.fillna(self.df.median(), inplace=True)
            return self.df
        
        except Exception as e:
            CustomException(e, sys)

    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Divides the data into train and test datasets"""

        try:
            X = df.drop("math_score", axis=1)
            y = df["math_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            CustomException(e, sys)

