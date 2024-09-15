import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client


client = Client()
active_stack = client.active_stack
# print(f"Active stack: {active_stack.name}")
experiment_tracker = active_stack.experiment_tracker
# Add MLflow experiment tracker to the active stack

if experiment_tracker is None:
    raise Exception("Experiment tracker not found in the active stack.")
else:
    print(f"Experiment tracker found: {experiment_tracker.name}")

@step(experiment_tracker=experiment_tracker.name)

def train_model(x_train:pd.DataFrame,x_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,config:ModelNameConfig)->RegressorMixin:
    """
    Trains the Linear Regression model.

    Args:
        x_train (pd.DataFrame): The training features.
        x_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.

    Returns:
        LinearRegressionModel: The trained Linear Regression model.
    """
    model=None
    if config.model_name=="Linear Regression":
        mlflow.sklearn.autolog()
        model=LinearRegressionModel()
        trained_model=model.train(x_train,y_train)
        return trained_model
    else:
        logging.error("Model not found")
        raise Exception("Model not found")