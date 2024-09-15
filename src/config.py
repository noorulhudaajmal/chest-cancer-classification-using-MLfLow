from src.config_manager import DataIngestionConfig
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
