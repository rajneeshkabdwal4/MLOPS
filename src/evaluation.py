import logging
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class Evaluation:
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method.")

class MSE(Evaluation):
    """
    Class to calculate Mean Squared Error (MSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Mean Squared Error.
        """
        try:
            logging.info("Calculating Mean Squared Error.")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error occurred in calculating Mean Squared Error: {e}")
            raise e

class R2(Evaluation):
    """
    Class to calculate R2 Score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the R2 Score between true and predicted values.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: R2 Score.
        """
        try:
            logging.info("Calculating R2 Score.")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error occurred in calculating R2 Score: {e}")
            raise e

class RMSE(Evaluation):
    """
    Class to calculate Root Mean Squared Error (RMSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: Root Mean Squared Error.
        """
        try:
            logging.info("Calculating Root Mean Squared Error.")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error occurred in calculating Root Mean Squared Error: {e}")
            raise e