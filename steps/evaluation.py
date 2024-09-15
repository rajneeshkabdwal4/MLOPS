import logging
from zenml import step
from src.evaluation import MSE, R2, RMSE
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml.client import Client

import mlflow
experiment_tracker=Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   x_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float, "R2 Score"],
                       Annotated[float, "RMSE"],
                       Annotated[float, "MSE"]
                   ]:
    
    """
    Evaluates a regression model using R2 Score, RMSE, and MSE.

    Args:
        model (RegressorMixin): The regression model to evaluate.
        x_test (pd.DataFrame): The test features.
        y_test (pd.DataFrame): The test labels.

    Returns:
        Tuple[Annotated[float, "R2 Score"], Annotated[float, "RMSE"], Annotated[float, "MSE"]]:
            A tuple containing the R2 Score, RMSE, and MSE of the model.
    """
    try:
        logging.info('Evaluating model...')
        # Make predictions
        predictions = model.predict(x_test)
        # Calculate evaluation metrics
        mse = MSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("MSE", mse)
        r2 = R2().calculate_scores(y_test, predictions)
        mlflow.log_metric("R2", r2)
        rmse = RMSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("RMSE", rmse)
        
        return r2, rmse, mse
    except Exception as e:
        logging.error(f"Error occurred in evaluating model: {e}")
        raise e