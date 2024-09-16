from src import logger
from src.config import ConfigManager
from src.base_model import BaseModel

STAGE_NAME = "Model Initialization Step"


def model_initialization_step(config: ConfigManager):
    """
    Prepare base model based on saved
    configuration and model params
    """
    logger.info(f">>> {STAGE_NAME} started.")

    base_model_config = config.get_basemodel_config()
    base_model = BaseModel(config=base_model_config)
    base_model.get_base_model()
    base_model.update_base_model()

    logger.info(f">>> {STAGE_NAME} completed.")


