from azureml.core import Workspace
from azureml.core.model import Model

model_name = "Grocery-Prediction"
endpoint_name = "prediction-ep"

ws = Workspace.from_config()

# Locate the model in the workspace
model = Model(ws, name=model_name)

# Deploy the model as a real-time endpoint
service = Model.deploy(ws, endpoint_name, [model])

# Wait for the model deployment to complete
service.wait_for_deployment(show_output=True)