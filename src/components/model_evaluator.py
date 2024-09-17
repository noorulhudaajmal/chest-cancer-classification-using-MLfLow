import tensorflow as tf

from src import logger
from src.components.data_preprocessor import DataPreprocessor
from src.config.config_manager import ModelEvaluationConfig, DataPreprocessingConfig
from pathlib import Path



class ModelEvaluator:
    """
    Evaluates the trained model on test dataset.
    """

    def __init__(self, eval_config: ModelEvaluationConfig, preprocessing_config: DataPreprocessingConfig):
        self.score = None
        self.config = eval_config
        self.data_preprocessor = DataPreprocessor(config=preprocessing_config)
        self.model = None
        self.test_generator = None


    def get_score(self):
        """
        Getter function for evaluation scores
        """
        return {"test_loss": self.score[0], "test_accuracy": self.score[1]}


    def process_test_data(self):
        """
        Use the preprocessor to get the test data generator
        """
        logger.info("Preprocessing test data for evaluation.")
        self.test_generator = self.data_preprocessor.preprocess_test_data()

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads keras Model from specified location.

        :param path: filepath to saved model.
        :return: keras Model instance.
        """
        logger.info(f"Loading model from {path}.")
        return tf.keras.models.load_model(path)

    def evaluate_model(self):
        """
        Evaluates the models by testing it on test-data
        and save the evaluation score.
        """
        if self.test_generator is None:
            raise ValueError("Test data has not been preprocessed. Call preprocess_test_data() before evaluation.")

        logger.info("Starting model evaluation on test data.")
        self.model = self.load_model(self.config.model_path)
        self.score = self.model.evaluate(self.test_generator)

        scores = {"test_loss": self.score[0], "test_accuracy": self.score[1]}
        logger.info(f"Model Scores: {scores}.")


