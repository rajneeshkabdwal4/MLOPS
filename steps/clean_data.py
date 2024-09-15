import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataDivideStrategy,DataCleaning
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "x_train"], Annotated[pd.DataFrame, "x_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
    """
    Cleans and processes the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned and processed.

    Returns:
        Tuple[Annotated[pd.DataFrame, "x_train"], Annotated[pd.DataFrame, "x_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
            A tuple containing the training features, testing features, training labels, and testing labels.
    """
    try:
        # Data Preprocessing
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        # logging.info("Data preprocessing completed successfully.")

        # Data Division
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning successful.")

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error occurred in data cleaning: {e}")
        raise e
