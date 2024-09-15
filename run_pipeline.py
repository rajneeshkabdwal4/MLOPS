from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
if __name__=="__main__":
    # Run pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline("olist_customers_dataset.csv")

# mlflow ui --backend-store-uri file:"C:\Users\rajne\AppData\Roaming\zenml\local_stores\c3836c1e-be25-4075-861f-4c7f75eaeb34\mlruns"