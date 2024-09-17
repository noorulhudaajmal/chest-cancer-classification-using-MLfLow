import mlflow

from src import logger
from src.config.config import ConfigManager
from src.components.model_evaluator import ModelEvaluator

STAGE_NAME = "Model Evaluation Step"


def model_evaluation_step(config: ConfigManager):
    """
    Evaluates the trained model using ModelEvaluator.
    """
    logger.info(f">>> {STAGE_NAME} started.")

    model_evaluation_config = config.get_model_evaluation_config()
    data_preprocessing_config = config.get_data_preprocessing_config()
    model_evaluator = ModelEvaluator(eval_config=model_evaluation_config,
                                     preprocessing_config=data_preprocessing_config)
    model_evaluator.process_test_data()
    model_evaluator.evaluate_model()

    mlflow.log_metrics(model_evaluator.get_score())

    logger.info(f">>> {STAGE_NAME} completed.")


if __name__ == "__main__":
    config = ConfigManager()
    model_evaluation_step(config)