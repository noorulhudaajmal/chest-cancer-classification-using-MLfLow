import json
import os
from pathlib import Path
from src import logger
import yaml


def read_yaml(file_path: Path) -> dict:
    """
    Reads in yaml file.

    :param file_path: path to yaml file/

    :return: yaml params
    """
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file from {file_path} loaded successfully.")
            return content
    except Exception as e:
        logger.info(f"Error occurred while opening yaml file: {e}")
        raise e


def create_directories(dirs_path: list):
    """
    Create directories.

    :param dirs_path: list of paths of directories to be created.
    """
    for path in dirs_path:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created at {path}.")


def save_json(path: Path, data: dict):
    """
    save json data
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}.")


def load_json(path: Path) -> dict:
    """
    load json files data
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded successfully from: {path}.")
    return content
