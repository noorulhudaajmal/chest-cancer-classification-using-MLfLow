from src import logger
from src.config import ConfigManager
from src.base_model import BaseModel
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer

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

    eval_params = {
        "score": model_evaluator.score
    }
    logger.info(f">>> {STAGE_NAME} completed.")

    return eval_params