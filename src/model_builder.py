import tensorflow as tf
from abc import ABC, abstractmethod



class KerasCNNModel(ABC):
    """
    Abstract class for Keras pretrained CNN Models.
    """
    @abstractmethod
    def create_model(self, include_top: bool, weights: str, input_img_size: list) -> tf.keras.Model:
        """
        Creates model using keras application model api.

        :param include_top: adding 3 fully connected layers at top of network.
        :param weights: type of weights to use for training (None for random initialization).
        :param input_img_size: dimensions of input image.
        :return: tensorflow keras Model instance.
        """
        pass


class VGG16Model(KerasCNNModel):
    """
    Concrete implementation of
    KerasCNNModel for VGG16 Model
    """

    def create_model(self, include_top: bool, weights: str, input_img_size: list) -> tf.keras.Model:
        """
        Creates VGG16 using keras application model api.

        :param include_top: adding 3 fully connected layers at top of network.
        :param weights: type of weights to use for training (None for random initialization).
        :param input_img_size: dimensions of input image.
        :return: tensorflow keras VGG16 Model instance.
        """
        return tf.keras.applications.VGG16(
            include_top=include_top,
            weights=weights,
            input_shape=input_img_size
        )


class MobileNetModel(KerasCNNModel):
    """
    Concrete implementation of
    KerasCNNModel for MobileNet Model
    """

    def create_model(self, include_top: bool, weights: str, input_img_size: list) -> tf.keras.Model:
        """
        Creates MobileNet using keras application model api.

        :param include_top: adding 3 fully connected layers at top of network.
        :param weights: type of weights to use for training (None for random initialization).
        :param input_img_size: dimensions of input image.
        :return: tensorflow keras MobileNet Model instance.
        """
        return tf.keras.applications.MobileNet(
            include_top=include_top,
            weights=weights,
            input_shape=input_img_size
        )


class ResNet50Model(KerasCNNModel):
    """
    Concrete implementation of
    KerasCNNModel for ResNet50 Model
    """

    def create_model(self, include_top: bool, weights: str, input_img_size: list) -> tf.keras.Model:
        """
        Creates ResNet50 using keras application model api.

        :param include_top: adding 3 fully connected layers at top of network.
        :param weights: type of weights to use for training (None for random initialization).
        :param input_img_size: dimensions of input image.
        :return: tensorflow keras ResNet50 Model instance.
        """
        return tf.keras.applications.ResNet50(
            include_top=include_top,
            weights=weights,
            input_shape=input_img_size
        )


class ModelFactory(ABC):
    @staticmethod
    def get_cnn_model(model_type: str) -> KerasCNNModel:
        """
        Provides appropriate keras Model based
        on specified model type.
        """
        if model_type == "vgg16":
            return VGG16Model()
        elif model_type == "mobilenet":
            return MobileNetModel()
        elif model_type == "resnet50":
            return ResNet50Model()
        else:
            raise ValueError(f"Model {model_type} is not supported.")


