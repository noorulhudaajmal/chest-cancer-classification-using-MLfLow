from urllib.parse import urlparse

import mlflow

from src import logger
from src.config import ConfigManager
from steps.data_ingestion_step import data_ingestion_step
from steps.model_evaluation_step import model_evaluation_step
from steps.model_preparation_step import model_initialization_step
from steps.model_training_step import model_training_step



def mlflow_pipeline():
    """
    End to end mlflow pipeline for training a CNN classifier
    """

    logger.info("Loading configuration.")
    config_manager = ConfigManager()

    # mlflow_config = config_manager.get_mlflow_config()

    # mlflow.set_registry_uri(mlflow_config.mlflow_uri)
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Set the experiment name
    mlflow.set_experiment("Chest Cancer Classification")

    with mlflow.start_run(run_name="VGG16") as run:
        run_id = run.info.run_id

        mlflow.set_tag("model_type", "cnn")
        mlflow.set_tag("author", "Huda")
        mlflow.set_tag("version", "1.0")
        mlflow.set_tags({
            "data_version": "v1.0",
            "project": "chest_cancer_classification",
            "purpose": "training model with keras VGG16"
        })
        # Log experiment description
        mlflow.set_tag("description", "An experiment to classify chest cancer.")

        logger.info(f"Pipelines initiated...")

        mlflow.log_params(config_manager.params)

        # 1. Data Ingestion
        data_ingestion_step(config=config_manager)

        # 2. Model Preparation step
        model_initialization_step(config=config_manager)

        # 3. Model Training Step
        model_training_step(config=config_manager)

        # 4. Model Evaluation Step
        model_evaluation_step(config=config_manager)

        logger.info(f"Pipeline operations completed successfully.")



if __name__ == "__main__":
    mlflow_pipeline()
