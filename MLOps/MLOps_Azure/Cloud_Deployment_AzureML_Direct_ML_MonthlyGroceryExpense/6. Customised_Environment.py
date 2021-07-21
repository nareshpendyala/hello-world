# Custom Deployment Environment Script

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

environment = Environment("my-custom-environment")

environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults>= 1.0.45', # mandatory dependency, contains the functionality needed to host the model as a web service
    'inference-schema[numpy-support]', # dependency for automatic schema generation (for parsing and validating input data)
    'joblib',
    'numpy',
    'scikit-learn'
])

# Inference Configuration Code

from azureml.core.model import InferenceConfig
inference_config = InferenceConfig(entry_script="score.py", environment=environment)

# Deployment Configuration Code

from azureml.core.webservice import AciWebservice
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)

# Deploy the model on customised environment

from azureml.core import Workspace
from azureml.core.model import Model

model_name = "Grocery-Prediction"
endpoint_name = "grocery-prediction-ep1"

ws = Workspace.from_config()

model = Model(ws, name=model_name)

service = Model.deploy(workspace=ws,
                       name=endpoint_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)

service.wait_for_deployment(show_output=True)
