import os
from pathlib import Path

from src.config_manager import DataIngestionConfig, BaseModelConfig, ModelTrainingConfig, DataPreprocessingConfig, \
    ModelEvaluationConfig, MLFlowConfig
from src.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from src.utils import read_yaml, create_directories


class ConfigManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAM_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])


    def get_dataingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            source=config["data_config"]["source"],
            sourceURL=config["data_config"]["sourceURL"],
            username=config["data_config"]["username"],
            extract_to=config["data_config"]["extract_to"]
        )

        return data_ingestion_config


    def get_basemodel_config(self) -> BaseModelConfig:
        config = self.config["base_model"]

        create_directories([config["root_dir"]])

        base_model_config = BaseModelConfig(
            root_dir=Path(config["root_dir"]),
            base_model_path=Path(config["base_model_path"]),
            updated_base_model_path=Path(config["updated_base_model_path"]),
            model_type=self.params["MODEL_TYPE"],
            input_img_size=self.params["IMAGE_SIZE"],
            params_lr=self.params["LEARNING_RATE"],
            include_top=self.params["INCLUDE_TOP"],
            weights=self.params["WEIGHTS"],
            classes=self.params["CLASSES"],
            optimizer=self.params["OPTIMIZER"],
            loss_function=self.params["LOSS_FUNCTION"]
        )

        return base_model_config


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        model_params = self.params
        training_data_dir = os.path.join(self.config["data_ingestion"]["data_config"]["extract_to"], "Data")

        model_training_config = DataPreprocessingConfig(
            training_data=Path(training_data_dir),
            batch_size=model_params["BATCH_SIZE"],
            is_augmentation=model_params["AUGMENTATION"],
            img_size=model_params["IMAGE_SIZE"]
        )

        return model_training_config


    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config["model_training"]
        base_model_config = self.config["base_model"]
        model_params = self.params
        training_data_dir = os.path.join(self.config["data_ingestion"]["data_config"]["extract_to"], "Data")

        create_directories([training_config["root_dir"]])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(training_config["root_dir"]),
            trained_model_path=Path(training_config["trained_model_path"]),
            base_model_path=Path(base_model_config["updated_base_model_path"]),
            training_data=Path(training_data_dir),
            n_epochs=model_params["EPOCHS"],
            batch_size=model_params["BATCH_SIZE"],
            is_augmentation=model_params["AUGMENTATION"],
            img_size=model_params["IMAGE_SIZE"]
        )

        return model_training_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        training_config = self.config["model_training"]
        training_data_dir = os.path.join(self.config["data_ingestion"]["data_config"]["extract_to"], "Data")

        create_directories([training_config["root_dir"]])

        model_evaluation_config = ModelEvaluationConfig(
            model_path=Path(training_config["trained_model_path"]),
            training_data=Path(training_data_dir),
            all_params=self.params,
        )

        return model_evaluation_config


    def get_mlflow_config(self) -> MLFlowConfig:
        mlflow_config = self.config["mlflow"]
        return MLFlowConfig(
            mlflow_uri=mlflow_config["mlflow_uri"]
        )


