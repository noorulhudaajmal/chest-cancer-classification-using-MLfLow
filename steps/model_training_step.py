import mlflow

from src import logger
from src.config.config import ConfigManager
from src.components.model_trainer import ModelTrainer

STAGE_NAME = "Model Training Step"


def model_training_step(config: ConfigManager):
    """
    Prepare base model based on saved
    configuration and model params
    """
    logger.info(f">>> {STAGE_NAME} started.")

    model_training_config = config.get_model_training_config()
    data_preprocessing_config = config.get_data_preprocessing_config()
    model_trainer = ModelTrainer(train_config=model_training_config,
                                 preprocessing_config=data_preprocessing_config)
    model_trainer.get_base_model()
    model_trainer.preprocess_data()
    mlflow.tensorflow.autolog(log_datasets=False)
    history = model_trainer.train()

    logger.info(f">>> {STAGE_NAME} completed.")


if __name__ == "__main__":
    config = ConfigManager()
    model_training_step(config)