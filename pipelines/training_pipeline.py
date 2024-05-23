# Pipeline for model training, validation, and saving
import sys
import os

from src.logger import logger
from src.exceptions import CustomException

from src.train_model import train_model
from src.evaluation import evaluation
from pipelines.data_pipeline import student_performance_data_pipeline
from src.utils import save_object_pickle, load_object

class ModelTrainerConfig:
    """Model trainer config class"""
    trained_model_file_path = os.path.join('data', 'processed', 'model.pkl')
    model_score_threshold = 0.80

def student_performance_training_pipeline(model_type: str ="randomforest"):
    """Pipeline for model training, validation, and saving"""
    model_trainer_config=ModelTrainerConfig()

    try:
        logger.info("Training pipeline for student performance started")
        train_array, test_array, preprocessor_obj = student_performance_data_pipeline()

        X_train = train_array[:, : -1]
        y_train = train_array[:, -1]
        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]
         
        print(X_train.shape)
        sklearn_regressor_model = train_model(X_train, y_train, X_test, y_test, model_type=model_type, do_fine_tuning=True)

        r2_score, mse, rmse = evaluation(sklearn_regressor_model, X_test, y_test)
               
        model_score = r2_score
        if model_score < model_trainer_config.model_score_threshold:
            raise CustomException(sys, f"Model score must be greater than the configured threshold: {model_trainer_config.model_score_threshold}")
        
        save_object_pickle(model_trainer_config.trained_model_file_path, sklearn_regressor_model)
        logger.info("Training pipeline for student performance completed successfully")
        return r2_score, mse, rmse

        
    except Exception as e:
        CustomException(e, sys)
    

if __name__ == '__main__':
    student_performance_training_pipeline(model_type="randomforest")