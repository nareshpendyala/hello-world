import joblib
import os
import json
import numpy as np

from azureml.core import Run
from azureml.core.model import Model


# The init() method is called once, when the web service starts up.
def init():
    """
    Initialize the scoring script
    """
    global model, run

    # Locate the model in Azure Machine Learning
    model_artifact = "groceryprediction.joblib"
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_artifact)
    # AZUREML_MODEL_DIR is an environment variable created during deployment,
    # indicating the path to the folder containing the registerd moedels
    # (./azureml-models/$MODEL_NAME/$VERSION)

    # Load the model
    model = joblib.load(model_path)


# The run() method is called each time a request is made to the scoring API.
def run(data):
    """
    Make predictions
    """

    # Extract the input data from the request
    # (we expect it to be a 2D array, where every row
    # is a different sample for the model to process)
    input_data = json.loads(data)['data']

    # Run the model on the input data
    output = model.predict(np.array(input_data))

    # Prepare the results as a dictionary,
    # so that it can be JSON-serialized
    # (as the returned data needs to be JSON-serializable)
    result = {"House-Price" : output.tolist()}

    return result