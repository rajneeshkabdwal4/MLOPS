from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

import pandas as pd
@pipeline(enable_cache=True)


def training_pipeline(data_path:str):
    # x=pd.read_csv(data_path)
    # print(x.head())
    print("Training pipeline")
    df=ingest_df(data_path)
    x_train,x_test,y_train,y_test=clean_df(df)
    model=train_model(x_train,x_test,y_train,y_test)
    r2,rmse,mse=evaluate_model(model,x_test,y_test)
    