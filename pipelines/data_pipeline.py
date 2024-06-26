# Pipeline for data loading, cleaning, and preprocessing

import sys
from typing import Annotated, Tuple

import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src import ingest_data, clean_data, transform_data
from src.logger import logger

def student_performance_data_pipeline() -> Tuple[
        Annotated[np.ndarray, "train_array"],
        Annotated[np.ndarray, "test_array"],
        Annotated[str, "preporssor object"],
    ]:
    """
    Pipeline for data wraggling, cleaning
    Steps in this pipeline:
    1. Load data from the data source:  csv, api, cloud, etc...
        call ingest_data.py
    2. Clean data
        call clean_data.py
    3. Transform data
        call feature_engineering.py

    Args:
        None

    Returns:
        X_train, y_train, y_test, x_train, x_test (pd.DataFrame, pd.Series) >> Cleansed, transformed split dataset ready for model training and evaluation
    """
  
    try:        
        logger.info("Data pipeline for student performance started")
        # Step 1: Load data from the data source:  csv, api, cloud, etc...
        data_injection = ingest_data.DataIngestion()
        raw_data_path, train_data_path, test_data_path = data_injection.initiate_data_ingestion()
        print(f"train_data_path: {train_data_path}")
        print(f"test_data_path: {test_data_path}")
         

        # Step 2: Clean data        
        data_cleaning = clean_data.DataCleaning(raw_data_path, train_data_path, test_data_path)
        data_cleaning.clean_data()

        # Step 3: Transform data
        data_transformation = transform_data.DataTransformation()
        train_array, test_array, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        logger.info("Data pipeline for student performance completed successfully")
        return (train_array, test_array, preprocessor_obj_file_path)
    
    except Exception as e:
        CustomException(e, "Error in data pipeline")


if __name__ == '__main__':
    student_performance_data_pipeline()