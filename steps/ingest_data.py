import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self)->None:
        pass
    
    def get_data(self):
        df = pd.read_csv("olist_customers_dataset.csv")
        return df
    
@step
def ingest_df(data_path : str) -> pd.DataFrame:
    try:
        ingest_data = IngestData()
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in IngestData step: {e}")
        return None
