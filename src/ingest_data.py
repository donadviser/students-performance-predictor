import sys
import os
from src.logger import logger
from src.exceptions import CustomException
from src.utils import get_config_data

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple


@dataclass
class DataIngestionConfig:
    original_data_path: str = os.path.join("data", "raw", "stud.csv")
    raw_data_path: str = os.path.join('data', "processed", "data.csv")
    train_data_path: str = os.path.join('data', "processed", "train.csv")
    test_data_path: str = os.path.join('data', "processed", "test.csv")

class DataIngestion:
    """Data ingestion class which ingests from the souce and returns a DataFrame"""

    def __init__(self)-> None:
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """Returns the raw data"""
        logger.info("Get original raw data")    

        try:
            original_data_path = self.ingestion_config.original_data_path
            print(f"original_data_path: {original_data_path}")
            df = pd.read_csv(original_data_path) 
            print(df.head())
            logger.info("orignal raw data read to DataFrame successfully")
            raw_data_path = self.ingestion_config.raw_data_path
            
            #raw_data_path = os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            print(f"raw_data_path: {raw_data_path}")
            df.to_csv(raw_data_path, index=False, header=True)
            logger.info(f"raw data saved successfully to: {raw_data_path}")

            logger.info("Initiate data split to train and test set")
            train_data_path = self.ingestion_config.train_data_path    
            test_data_path = self.ingestion_config.test_data_path           
             
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(train_data_path, index=False, header=True)
            test_set.to_csv(test_data_path, index=False, header=True)
            logger.info(f"train and test data saved successfully to: {train_data_path} and {test_data_path}")
            logger.info("Ingestion of data is completed successfully")           

            return(
                train_data_path,
                test_data_path               
            )
        except Exception as e:
            raise CustomException(e, '')

if __name__ == '__main__':
    data_injection = DataIngestion()
    train_data_path, test_data_path = data_injection.initiate_data_ingestion()
    print(f"train_data_path: {train_data_path}")
    print(f"test_data_path: {test_data_path}")