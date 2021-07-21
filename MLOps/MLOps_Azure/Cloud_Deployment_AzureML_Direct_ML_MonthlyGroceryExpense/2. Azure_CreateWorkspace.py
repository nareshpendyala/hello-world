# pip install --upgrade azureml-sdk
from azureml.core import Workspace

NAME = "aml-test-ws" # The name of the workspace to be created
RESOURCE_GROUP = "Analytics-4" # The name of the resource group that will contain the workspace
LOCATION = "westeurope" # The data center location to deploy to
SUBSCRIPTION_ID = "<your-subscription-id>" # GUID identifying the subscription with which to deploy

ws = Workspace.create(
    name=NAME,
    resource_group=RESOURCE_GROUP, 
    subscription_id=SUBSCRIPTION_ID,
    location=LOCATION,
    sku='basic', # Azure ML version (available: basic or enterprise)
    create_resource_group=True, # A new workspace will be created if it doesn't exist yet
    exist_ok=True, # The method succeeds if the workspace already exists
    show_output=True)

ws.write_config()