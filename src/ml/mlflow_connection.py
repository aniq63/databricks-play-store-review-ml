import mlflow
import os
import sys
from dotenv import load_dotenv
from src.logger import logging
from src.exception import MyException

class MLflowConnection:
    """
    Manages connection to MLflow experiments on Databricks from local environment.
    """
    def __init__(self):
        """
        Initialize and load environment variables.
        """
        load_dotenv()
        self.experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")

    def connect(self):
        """
        Establish connection to the specified MLflow experiment.
        """
        logging.info("Attempting to connect to MLflow...")
        try:
            if not self.experiment_id:
                raise ValueError("MLFLOW_EXPERIMENT_ID is missing in .env file")

            # Set tracking URI (default to 'databricks')
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set the experiment
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
            logging.info(f"Successfully connected to MLflow experiment: {self.experiment_id}")
            print(f"Connected to MLflow experiment: {self.experiment_id}")

        except Exception as e:
            logging.error(f"Failed to connect to MLflow: {e}")
            raise MyException(e, sys)

def setup_mlflow():
    """
    Helper function to initialize and connect to MLflow.
    """
    conn = MLflowConnection()
    conn.connect()