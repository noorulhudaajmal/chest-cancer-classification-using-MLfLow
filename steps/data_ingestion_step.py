from src import logger
from src.config.config import ConfigManager
from src.components.data_ingestor import DataIngestorFactory


STAGE_NAME = "Data Ingestion Step"


def data_ingestion_step(config: ConfigManager):
    """
    Ingests data based on stored configuration
    """
    logger.info(f">>> {STAGE_NAME} started.")

    data_ingestion_config = config.get_dataingestion_config()
    data_ingestor = DataIngestorFactory.get_data_ingestor(data_ingestion_config)
    data_ingestor.ingest()

    logger.info(f">>> {STAGE_NAME} completed.")



if __name__ == "__main__":
    config = ConfigManager()
    data_ingestion_step(config)