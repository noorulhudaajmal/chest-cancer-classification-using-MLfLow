from keras.src.legacy.preprocessing.image import ImageDataGenerator

from src import logger
from src.config_manager import DataPreprocessingConfig


class DataPreprocessor:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    @staticmethod
    def create_image_data_generator(augmentation: bool) -> ImageDataGenerator:
        if augmentation:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                validation_split=0.20
            )
        else:
            return ImageDataGenerator(
                rescale=1./255,
                validation_split=0.20
            )

    def preprocess_data(self):
        logger.info(f"Initializing training and validation data generators.")
        # creating generators
        train_gen = self.create_image_data_generator(augmentation=self.config.is_augmentation)
        valid_gen = self.create_image_data_generator(augmentation=False)

        #preparing training data generator
        train_generator = train_gen.flow_from_directory(
            directory=self.config.training_data / 'train',
            # subset="training",
            shuffle=True,
            target_size=self.config.img_size[:-1],
            batch_size=self.config.batch_size,
            interpolation="bilinear"
        )
        logger.info(f"Training Data processing completed.")

        # preparing validation data generator
        validation_generator = valid_gen.flow_from_directory(
            directory=self.config.training_data / 'valid',
            # subset="validation",
            shuffle=False,
            target_size=self.config.img_size[:-1],
            batch_size=self.config.batch_size,
            interpolation="bilinear"
        )
        logger.info(f"Validation Data processing completed.")

        return train_generator, validation_generator


    def preprocess_test_data(self):
        logger.info(f"Initializing testing data generator.")
        # Preprocessing test data without augmentation
        test_gen = ImageDataGenerator(rescale=1./255)

        # Test data generator
        test_generator = test_gen.flow_from_directory(
            directory=self.config.training_data / 'test',
            shuffle=False,
            target_size=self.config.img_size[:-1],
            batch_size=self.config.batch_size,
            interpolation="bilinear"
        )
        logger.info(f"Testing Data processing completed.")

        return test_generator