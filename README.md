# chest-cancer-classification-using-MLfLow
This project is designed for chest cancer classification using the keras Model API and MLflow for experiment tracking. The project utilizes data from [kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data) and provides flexibility in model configuration and training parameters through YAML configuration files.

## Project Overview

The main components of this project include:

- **Data Ingestion**: Handles data ingestion from local sources, Kaggle, or Google Drive.
- **Model Training**: Utilizes a keras pre-trained models for classification.
- **MLflow Integration**: Tracks experiments, model parameters, and performance metrics.
- **Streamlit app**: Streamlit app to use trained model for prediction on custom data.

## Installation:

1. Clone the repository:
```shell
  git clone https://github.com/noorulhudaajmal/chest-cancer-classification-using-MLfLow.git
  cd chest-cancer-classification-using-MLfLow
```

2. Create a virtual environment and install dependencies:
```shell
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  pip install -r requirements.txt
```

## Usage

- Update the config.yaml file to specify data scouce:
  - Supported methods are `local`, `kaggle`, `gdrive`.
  - For local, update the sourceURL to zipped file location.
  - For kaggle, update sourceURL to kaggle dataset, set username to kaggle username.
  - For gdrive, update the sourceURL to google drive file ID with public access.

- params.yaml can be adjusted accordingly to model choice.
  - Supported MODEL_TYPE choices are `vgg16`, `mobilenet` & `resnet50`.
  - Supported LOSS_FUNCTION choices are `categorical_crossentropy`, `binary_crossentropy`, `mean_squared_error` & `sparse_categorical_crossentropy`.
  - Supported OPTIMIZER choices are `sgd`, `adam` & `rmsprop`.
  - Set the IMAGE_SIZE parameter to be according to selected model and input images.

- Train the model and log experiments with MLflow:
    ```shell
    python main.py
    ```

- Run the streamlit app for model inferencing:
   ```shell
      streamlit run app.py
   ```
  Access the Streamlit app at http://localhost:8501.



