from pathlib import Path
from src import logger

import tensorflow as tf
from src.config.config_manager import ModelTrainingConfig, DataPreprocessingConfig
from src.components.data_preprocessor import DataPreprocessor


class ModelTrainer:
    def __init__(self, train_config: ModelTrainingConfig, preprocessing_config: DataPreprocessingConfig):
        logger.info(f"Model trainer initiated.")
        self.config = train_config
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.data_preprocessor = DataPreprocessor(config=preprocessing_config)

    def get_model(self):
        """
        Getter method to retrieve the current model.

        :return: The model instance.
        """
        return self.model

    def get_base_model(self):
        logger.info(f"Loading base model from {self.config.base_model_path}.")
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )

    def preprocess_data(self):
        logger.info(f"Preprocessing data before model training.")
        self.train_generator, self.validation_generator = self.data_preprocessor.preprocess_data()

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        logger.info(f"Saving the trained model to {path}.")
        model.save(path)


    def train(self):
        if self.train_generator is None or self.validation_generator is None:
            raise ValueError("Data has not been preprocessed. Call preprocess_data() before training.")

        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.validation_generator.samples // self.validation_generator.batch_size

        logger.info(f"Model training started with Epochs={self.config.n_epochs}.")
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.n_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.validation_generator
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)

        return history