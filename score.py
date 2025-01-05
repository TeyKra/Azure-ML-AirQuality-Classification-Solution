import logging
import time
import json
import numpy as np
import joblib
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.trace import tracer as tracer_module
from opencensus.trace.samplers import ProbabilitySampler
import mlflow

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add Azure Application Insights handler if the connection string is available
app_insights_connection_string = os.getenv("APPINSIGHTS_CONNECTION_STRING", "")
if app_insights_connection_string:
    azure_handler = AzureLogHandler(connection_string=app_insights_connection_string)
    logger.addHandler(azure_handler)
    logger.info("Application Insights handler added successfully.")
else:
    logger.warning("No Application Insights Connection String found.")

# Configure the tracer with a probability sampler (100% sampling rate)
tracer = tracer_module.Tracer(sampler=ProbabilitySampler(1.0))

# Mapping of class indices to their respective labels
CLASS_MAPPING = {
    0: "Good",
    1: "Hazardous",
    2: "Moderate",
    3: "Poor"
}

# Global variable to hold the loaded model
model = None

def init():
    """
    Initializes the model by loading it from the specified directory.
    This function should be called once when the application starts.
    """
    global model
    try:
        # Retrieve the model directory from environment variables
        model_root = os.getenv("AZUREML_MODEL_DIR")
        if not model_root:
            raise EnvironmentError("AZUREML_MODEL_DIR environment variable is not set.")
        
        # Construct the full path to the model file
        model_path = os.path.join(model_root, "mlflow_model", "model.pkl")
        
        # Load the model using joblib
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise

def run(data: str) -> dict:
    """
    Processes input data to make predictions using the loaded model.

    Parameters:
        data (str): JSON-formatted string containing the input data.

    Returns:
        dict: A dictionary with prediction results or an error message.
    """
    with tracer.span(name="PredictionRequest"):
        start_time = time.time()
        try:
            # Parse the input JSON data
            input_json = json.loads(data)
            logger.info(f"Received input data: {input_json}")

            # Extract the features matrix from the input data
            features = input_json.get("data")
            if features is None:
                raise KeyError("Missing 'data' field in input JSON.")

            # Validate the format of the input features
            if not isinstance(features, list) or not all(isinstance(row, list) and len(row) == 9 for row in features):
                raise ValueError("Invalid input format. Expecting a list of lists with 9 features each.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return {"error": "Invalid JSON format."}
        except (KeyError, ValueError) as e:
            logger.error(f"Input validation error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during input parsing: {e}")
            return {"error": "An unexpected error occurred while parsing input data."}

        try:
            # Record the time before making predictions
            prediction_start_time = time.time()

            # Make predictions using the loaded model
            predictions = model.predict(features)
            prediction_duration = time.time() - prediction_start_time

            # Calculate the total latency of the request
            total_latency = time.time() - start_time

            # Process all predictions
            formatted_predictions = []
            for pred in predictions:
                class_index = int(pred)
                class_label = CLASS_MAPPING.get(class_index, "Unknown")
                formatted_predictions.append(
                    f"prediction : class_index : {class_index}, class_label : \"{class_label}\", latency_seconds : {round(prediction_duration / len(predictions), 6)}"
                )

            # Log prediction details
            logger.info(
                f"Predictions made: {formatted_predictions}, "
                f"prediction_time={prediction_duration:.4f}s, total_latency={total_latency:.4f}s"
            )

            # Log metrics to MLflow
            mlflow.log_metric("prediction_latency", prediction_duration)
            mlflow.log_metric("total_latency", total_latency)
            mlflow.log_metric("input_data_size", len(features))

            # Log custom dimensions to Azure Application Insights
            logger.info({
                "custom_dimensions": {
                    "prediction_latency": prediction_duration,
                    "total_latency": total_latency,
                    "input_data_size": len(features),
                }
            })

            # Construct a structured response
            response = {
                "predictions": formatted_predictions,
                "total_latency_seconds": round(total_latency, 6)
            }
            return response
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": "An error occurred during prediction."}
