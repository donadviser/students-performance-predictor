import numpy as np
import pandas as pd
from typing import Annotated, Tuple

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

from src.exceptions import CustomException
from src.logger import logger

def evaluation(sklearn_regressor_model: RegressorMixin, 
               X_test: np.ndarray, 
               y_test: np.ndarray
               ) -> Tuple[
                   Annotated[float, "r2_score"],
                   Annotated[float, "mse: mean_squared_error"],
                   Annotated[float, "rmse: root_mean_squared_error"],
                   Annotated[float, "test_model_score"],
                   ]:
    """
    Evaluation of the Sklearn Regressor Model on the Test dataset and returns the metrics.

    Args:
        sklearn_regressor_model (RegressorMixin): Trained model
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels
        
    Returns:
        r2_score (float): metric like R-squared
        mse: mean_squared_error (float)
        rmse: root_mean_squared_error (float)
    """

    try:
        logger.info("Evaluating the model")
        y_pred = sklearn_regressor_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"The mean square error, mse value is: {mse}")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"The root mean square error, rmse value is: {rmse}")
        r2 = r2_score(y_test, y_pred)
        logger.info(f"The r2 score, r2 value is: {r2}")

        return r2, mse, rmse
    except Exception as e:
        raise CustomException(e, "Error in evaluation")
    