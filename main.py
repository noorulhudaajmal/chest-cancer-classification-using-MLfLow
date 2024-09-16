from src import logger
from src.config import ConfigManager
from steps.data_ingestion_step import data_ingestion_step
from steps.model_preparation_step import model_initialization_step

if __name__ == "__main__":
    logger.info(f"Pipelines initiated...")

    config_manager = ConfigManager()

    data_ingestion_step(config=config_manager)
    model_initialization_step(config=config_manager)

    logger.info(f"Pipeline operations completed successfully.")