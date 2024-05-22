# Pipeline for model training, validation, and saving
import sys

from src.logger import logger
from src.exceptions import CustomException

from src.train_model import train_model
from pipelines.data_pipeline import student_performance_data_pipeline


def student_performance_training_pipeline(model_type: str ="randomforest"):
    """Pipeline for model training, validation, and saving"""
    try:
        logger.info("Training pipeline for student performance started")
        X_train, X_test, y_train, y_test = student_performance_data_pipeline()
        print(f"shape of X_train: {X_train.shape} x_test: {X_test.shape}")
        print(f"shape of y_train: {y_train.shape} y_test: {y_test.shape}")

        sklearn_regressor_model = train_model(X_train, y_train, X_test, y_test, model_type=model_type, do_fine_tuning=True)
        


        logger.info("Training pipeline for student performance completed successfully")
    except Exception as e:
        CustomException(e, sys)
    

if __name__ == '__main__':
    student_performance_training_pipeline()