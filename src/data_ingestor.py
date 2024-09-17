import os
import zipfile
from src import logger
from abc import ABC, abstractmethod
import kaggle
import gdown
from gdown.exceptions import FileURLRetrievalError

from src.config_manager import DataIngestionConfig


class DataIngestor(ABC):
    """
    Abstract base class for data ingestion strategies.
    """

    @abstractmethod
    def ingest(self):
        """
        Ingest data based on class configuration.
        """
        pass


class LocalDataIngestor(DataIngestor):
    """
    Ingest data from a local file system.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initiates the local data ingestor strategy
        with specified configuration.

        :param config: strategy configuration
        """
        self.config = config
        # Validate configuration
        if self.config.sourceURL is None or self.config.extract_to is None:
            raise ValueError("LocalDataIngestor requires source directory path as 'sourceURL' and 'extract_to' directory path in config.")


    def ingest(self):
        """
        Ingest data from the source into the
        destination location
        """
        file_path = self.config.sourceURL
        extract_to = self.config.extract_to

        logger.info(f"Extracting data from {file_path} to {extract_to}.")

        # Extract data from zip file
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(extract_to)
        logger.info(f"Data extraction complete for {file_path}.")


class KaggleDataIngestor(DataIngestor):
    """
    Ingest data from Kaggle dataset using the Kaggle API.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initiates the kaggle data ingestor strategy
        with specified configuration.

        :param config: strategy configuration
        """
        self.config = config
        # Validate configuration
        if self.config.sourceURL is None or self.config.extract_to is None or self.config.username is None:
            raise ValueError("KaggleDataIngestor requires 'dataset URL', 'extract_to' directory, and 'username' in config.")

    def ingest(self):
        """
        Ingest data from kaggle dataset to
        destination location using kaggle api.
        """

        dataset = self.config.sourceURL
        extract_to = self.config.extract_to

        logger.info(f"Downloading Kaggle dataset {dataset}")

        # Authenticate and download dataset using Kaggle API
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=extract_to, unzip=True)

        logger.info(f"Kaggle dataset {dataset} downloaded and extracted to {extract_to}.")


class GoogleDriveDataIngestor(DataIngestor):
    """
    Ingest data from Google Drive using the gdown library.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initiates the g-drive data ingestor strategy
        with specified configuration.

        :param config: strategy configuration
        """
        self.config = config
        # Validate configuration
        if self.config.sourceURL is None or self.config.extract_to is None:
            raise ValueError("GoogleDriveDataIngestor requires 'sourceURL' and 'extract_to' in config.")


    def ingest(self):
        """
        Ingest data from Google Drive folder to
        destination location using gdown api.
        """
        gdrive_file_id = self.config.sourceURL
        extract_to = self.config.extract_to
        download_file_name = 'downloaded_data.zip'

        logger.info(f"Downloading data from Google Drive file ID {gdrive_file_id}.")

        try:
            # Google Drive URL for gdown
            gdrive_url = f"https://drive.google.com/uc?/export=download&id={gdrive_file_id}"
            output_path = os.path.join(extract_to, download_file_name)

            # Download file using gdown
            gdown.download(gdrive_url, output_path, quiet=False)

            logger.info(f"Google Drive file downloaded to {output_path}")
        except FileURLRetrievalError as e:
            logger.info(f"The file with ID: '{gdrive_file_id}' is not publicly accessible.")
            raise ValueError(f"Change the permissions of the file to publicly accessible. \n{e}")

        if output_path.endswith('.zip'):
            with zipfile.ZipFile(output_path, 'r') as zip_file:
                zip_file.extractall(extract_to)
            logger.info(f"Data extracted from Google Drive zip file to {extract_to}.")


class DataIngestorFactory:
    """
    Factory class to get the appropriate data ingestor.
    """

    @staticmethod
    def get_data_ingestor(config: DataIngestionConfig):
        """
        Factory method to select appropriate data ingestor based on config.

        :param config: A dictionary containing the required configuration.
        :return: An instance of a data ingestor class.
        """
        if config.source is None:
            raise ValueError("Config must contain 'source' to select data ingestor.")

        source = config.source

        if source == "local":
            return LocalDataIngestor(config)
        elif source == "kaggle":
            return KaggleDataIngestor(config)
        elif source == "gdrive":
            return GoogleDriveDataIngestor(config)
        else:
            raise ValueError(f"No ingestor available for the specified source: {source}.")



#mohamedhanyyy/chest-ctscan-images