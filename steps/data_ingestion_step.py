from src import logger
from src.config import ConfigManager
from src.injest_data import DataIngestorFactory


STAGE_NAME = "Data Ingestion Step"


def data_ingestion_step():
    """
    Ingests data based on stored configuration
    """
    logger.info(f">>> {STAGE_NAME} started.")

    config = ConfigManager()
    data_ingestion_config = config.get_dataingestion_config()
    data_ingestor = DataIngestorFactory.get_data_ingestor(data_ingestion_config)
    data_ingestor.ingest()

    logger.info(f">>> {STAGE_NAME} completed.")


data_ingestion_step()