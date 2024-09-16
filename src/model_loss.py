import tensorflow as tf
from abc import ABC, abstractmethod


class ModelLoss(ABC):
    """
    Abstract base class that defines the
    strategy interface for loss functions.
    """

    @abstractmethod
    def get_loss(self) -> tf.keras.losses.Loss:
        """
        Abstract method to initialize the tf.keras loss function.

        :return: an instance of a Keras loss function.
        """
        pass


class CategoricalCrossentropyLoss(ModelLoss):
    """
    Concrete strategy implementation for
    Categorical Crossentropy Loss.
    """

    def get_loss(self) -> tf.keras.losses.CategoricalCrossentropy:
        """
        Initializes the Categorical Crossentropy loss function.

        :return: an instance of Categorical Crossentropy loss.
        """
        return tf.keras.losses.CategoricalCrossentropy()


class BinaryCrossentropyLoss(ModelLoss):
    """
    Concrete strategy implementation
    for Binary Crossentropy Loss.
    """

    def get_loss(self) -> tf.keras.losses.BinaryCrossentropy:
        """
        Initializes the Binary Crossentropy loss function.

        :return: an instance of Binary Crossentropy loss.
        """
        return tf.keras.losses.BinaryCrossentropy()


class MeanSquaredErrorLoss(ModelLoss):
    """
    Concrete strategy implementation
    for Mean Squared Error Loss.
    """

    def get_loss(self) -> tf.keras.losses.MeanSquaredError:
        """
        Initializes the Mean Squared Error loss function.

        :return: an instance of Mean Squared Error loss.
        """
        return tf.keras.losses.MeanSquaredError()


class SparseCategoricalCrossentropyLoss(ModelLoss):
    """
    Concrete strategy implementation for
    Sparse Categorical Crossentropy Loss.
    """

    def get_loss(self) -> tf.keras.losses.SparseCategoricalCrossentropy:
        """
        Initializes the Sparse Categorical Crossentropy loss function.

        :return: an instance of Sparse Categorical Crossentropy loss.
        """
        return tf.keras.losses.SparseCategoricalCrossentropy()


class ModelLossFactory:
    """
    Factory class to get the appropriate loss
    function based on the specified choice.
    """

    @staticmethod
    def get_model_loss(loss_choice: str) -> ModelLoss:
        """
        Returns the appropriate loss function class based on the specified choice.

        :param loss_choice: A string representing the loss function choice.
                            Supported choices are "categorical_crossentropy",
                            "binary_crossentropy", "mean_squared_error",
                            "sparse_categorical_crossentropy".
        :return: An instance of ModelLoss.
        """
        if loss_choice == "categorical_crossentropy":
            return CategoricalCrossentropyLoss()
        elif loss_choice == "binary_crossentropy":
            return BinaryCrossentropyLoss()
        elif loss_choice == "mean_squared_error":
            return MeanSquaredErrorLoss()
        elif loss_choice == "sparse_categorical_crossentropy":
            return SparseCategoricalCrossentropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_choice}")
