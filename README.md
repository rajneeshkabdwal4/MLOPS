# MLOps Project

## Overview
This project demonstrates the implementation of an end-to-end MLOps pipeline. Using **ZenML**, the pipeline covers the entire machine learning lifecycle, from data ingestion and preprocessing to model training, deployment, and monitoring. The objective is to automate and streamline the workflow, ensuring better collaboration, reproducibility, and scalability.

## Project Structure

- `pipelines/`: Contains pipeline configurations.
- `src/`: Source code for model training and evaluation.
- `steps/`: Steps used in the pipeline (e.g., data preprocessing, training).
- `run_pipeline.py`: Script to execute the pipeline.
- `run_deployment.py`: Script to deploy the trained model.



## Setup

```bash
git clone https://github.com/rajneeshkabdwal4/MLOPS
pip install -r requirements.txt
zenml init
python run_pipeline.py
python run_deployment.py
```

## Dataset
The project uses the **olist_customers_dataset.csv** for customer segmentation. You can replace this dataset with your own by updating the relevant paths in the pipeline.

## Tools and Libraries
- **ZenML**: MLOps pipeline orchestration
- **Python**: Core language for development
- **Scikit-learn**: For machine learning models and metrics
- **Docker**: (Optional) For containerized deployments

## Pipeline Steps
1. **Data Ingestion**: Loads and prepares the dataset.
2. **Data Preprocessing**: Cleans and preprocesses the data for training.
3. **Model Training**: Trains a machine learning model using Scikit-learn.
4. **Evaluation**: Evaluates the trained model and logs performance metrics.
5. **Deployment**: Deploys the model for inference.

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.
