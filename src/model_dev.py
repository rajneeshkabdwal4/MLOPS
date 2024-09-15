import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    @abstractmethod

    def train(self, X, y):
        pass

    


class LinearRegressionModel(Model):
    def train(self, x_train, y_train,**kwargs):
        logging.info("Training Linear Regression model.")
        # Train the model
        try:
            reg=LinearRegression()
            reg.fit(x_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error occurred in training Linear Regression model: {e}")
            raise e