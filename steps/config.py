from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Configuration for the model name.
    """
    model_name: str = "Linear Regression"
    