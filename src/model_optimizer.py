import tensorflow as tf
from abc import ABC, abstractmethod


class ModelOptimizer(ABC):
    """
    Abstract base class that defines the
    strategy interface for optimizers.
    """

    @abstractmethod
    def get_optimizer(self, learning_rate: float) ->  tf.keras.optimizers.Optimizer:
        """
        Abstract method to initialize the tf keras
        optimizer with the specified learning rate.

        :param learning_rate: learning rate for the optimizer
        :return: an instance of keras optimizer.
        """
        pass


class SGDOptimizer(ModelOptimizer):
    """
    Concrete strategy implementation for SGD Optimizer.
    """
    def get_optimizer(self, learning_rate: float) -> tf.keras.optimizers.SGD:
        """
        Initializes the tf keras SGD optimizer
        with the specified learning rate.

        :param learning_rate: learning rate for the optimizer
        :return: an instance of keras SGD optimizer.
        """
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


class AdamOptimizer(ModelOptimizer):
    """
    Concrete strategy implementation for Adam Optimizer.
    """
    def get_optimizer(self, learning_rate: float) -> tf.keras.optimizers.Adam:
        """
        Initializes the tf keras Adam optimizer
        with the specified learning rate.

        :param learning_rate: learning rate for the optimizer
        :return: an instance of keras Adam optimizer.
        """
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class RMSPropOptimizer(ModelOptimizer):
    """
    Concrete strategy implementation for RMSProp Optimizer.
    """
    def get_optimizer(self, learning_rate: float) -> tf.keras.optimizers.RMSprop:
        """
        Initializes the tf keras RMSProp optimizer
        with the specified learning rate.

        :param learning_rate: learning rate for the optimizer
        :return: an instance of keras RMSProp optimizer.
        """
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


class ModelOptimizerFactory:
    """
    Factory class to get the appropriate optimizer
    strategy based on the specified choice.
    """
    @staticmethod
    def get_model_optimizer(optimizer_choice: str) -> ModelOptimizer:
        """
        Returns the appropriate optimizer class based
        on the specified optimizer choice.

        :param optimizer_choice:  A string representing the optimizer choice.
                                Supported choices are "sgd", "adam", "rmsprop".
        :return: An instance of ModelOptimizer.
        """
        if optimizer_choice == "sgd":
            return SGDOptimizer()
        elif optimizer_choice == "adam":
            return AdamOptimizer()
        elif optimizer_choice == "rmsprop":
            return RMSPropOptimizer()
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_choice}")
