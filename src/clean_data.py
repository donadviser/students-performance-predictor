import sys
import os
from typing import Annotated, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from src.exceptions import CustomException
from src.logger import logger


class DataCleaning:

    def __init__(self, raw_data_path: str, train_data_path: str, test_data_path: str) -> None:
        self.raw_data_path = raw_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def read_raw_data(self) -> pd.DataFrame:
        """Reads the raw data"""

        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info("raw data read to DataFrame successfully")
            return df
        except Exception as e:
            raise CustomException(e, "Error in clean_data during reading raw data")
         

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data: Removes columns which are not required, fills in missing values, and converts the data types"""

        try:
            #df = df.fillna(df.median(), inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e, "Error in clean_data during preprocess_data")

    def divide_data(self, df: pd.DataFrame) -> None:
        """Divides the data into train and test datasets"""

        try:
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            logger.info(f"train and test data saved successfully to: {self.train_data_path} and {self.test_data_path}")  
        except Exception as e:
            raise CustomException(e, "Error in clean_data during divide_data")


    def clean_data(self) -> None:
        """
        Data cleaning fuction which prepropocess the data and divides it into train and test data
        Args:
            data (pd.DataFrame): Data to be cleaned
            
        """
        try:
            logger.info("Cleaning data")
            df = self.read_raw_data()
            df = self.preprocess_data(df)
            self.divide_data(df)
        except Exception as e:
            raise CustomException(e, "Error in clean_data during clean_data")