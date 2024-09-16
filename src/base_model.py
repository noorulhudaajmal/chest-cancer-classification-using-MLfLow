from pathlib import Path
import tensorflow as tf
from src.config_manager import BaseModelConfig
from src.model_builder import ModelFactory
from src.model_loss import ModelLossFactory
from src.model_optimizer import ModelOptimizerFactory


class BaseModel:
    """
    Class to load and save base model
    """

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.updated_model = None
        self.classes = self.config.classes
        self.optimizer = ModelOptimizerFactory.get_model_optimizer(
            optimizer_choice=self.config.optimizer
        )
        self.learning_rate = self.config.params_lr
        self.loss_function = ModelLossFactory.get_model_loss(
            loss_choice=self.config.loss_function
        )


    def get_base_model(self):
        model_ins = ModelFactory.get_cnn_model(model_type=self.config.model_type)
        self.model = model_ins.create_model(
            include_top=self.config.include_top,
            weights=self.config.weights,
            input_img_size=self.config.input_img_size
        )
        self.save_model(path=self.config.base_model_path,
                        model=self.model)


    def prepare_model(self, freeze_all: bool, freeze_till: int) -> tf.keras.Model:
        """
        Prepares the base model by optionally freezing layers,
        adding custom classification layers, and compiling
        the final model.

        :param freeze_all: if True, freezes all layers in the model.
        :param freeze_till: number of layers from the end to keep trainable.
        :return: compiled model with additional configuration.
        """
        if freeze_all:
            for layer in self.model.layers:
                layer.trainable=False
        elif freeze_all is not None and freeze_till>0:
            for layer in self.model.layers[:-freeze_till]:
                layer.trainable=False

        flatten_in = tf.keras.layers.Flatten()(self.model.output)
        output = tf.keras.layers.Dense(
            units=self.classes,
            activation="softmax"
        )(flatten_in)

        prepared_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=output
        )

        prepared_model.compile(
            optimizer=self.optimizer.get_optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function.get_loss(),
            metrics=["accuracy"]
        )
        prepared_model.summary()

        return prepared_model


    def update_base_model(self):
        self.updated_model = self.prepare_model(
            freeze_all=True,
            freeze_till=None,
        )
        self.save_model(
            model=self.updated_model,
            path=self.config.updated_base_model_path
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)