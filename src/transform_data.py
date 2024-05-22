import sys
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logger
from src.utils import save_object_pickle

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data', 'processed', 'preprocessor.pkl')

class DataTransformation:
    """Data transformation class which transforms the data"""
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transform_object(self, numerical_columns: str, categorical_columns: str):
        """Returns the data transformation object"""
        try:
            logger.info("Get data transformation object")
            

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            

            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, "Error in transform_data while get_data_trasform_object")
        
    def initiate_data_transformation(self, train_path, test_path):
        """Initiates data transformation"""
        try:
            logger.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")
            

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Target column: {target_column}")
            logger.info(f"Shape of train dataframe: {train_df.shape}")
            logger.info(f"Shape of test dataframe: {test_df.shape}")
            
            preprocessor = self.get_data_transform_object(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logger.info("Applying preprocessing object on training and test dataframes.")


            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df)

            logger.info("concatenating preprocessed features and target variabled for boeth traing and test arrays")
            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logger.info("Saving preprocessor object")
            save_object_pickle(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path)
             
        except Exception as e:
            raise CustomException(e, "Error in transform_data while initiate_data_transformation")

