import logging 
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data: pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            xd = pd.read_csv("olist_customers_dataset.csv")
        except Exception as e:
            logging.error(f"Error occurred in reading data: {e}")
            raise e
        
        try:
            # Check if the columns exist before dropping them
            columns_to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            existing_columns = [col for col in columns_to_drop if col in data.columns]
            if not existing_columns:
                logging.warning("None of the specified columns exist in the DataFrame.")
            else:
                data = data.drop(existing_columns, axis=1)
            
            # Fill missing values with median or specific values
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error occurred in data processing: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        

class DataCleaning:
    def __init__(self, df, strategy):
        self.df = df
        self.strategy = strategy

    def handle_data(self):
        """
        Handles data cleaning and processing based on the provided strategy.

        Returns:
            Processed data based on the strategy.
        """
        data = None  # Initialize data to avoid UnboundLocalError
        try:
            data = self.strategy.handle_data(self.df)
            return data
        except Exception as e:
            logging.error(f"Error occurred in handle_data: {e}")
            raise e