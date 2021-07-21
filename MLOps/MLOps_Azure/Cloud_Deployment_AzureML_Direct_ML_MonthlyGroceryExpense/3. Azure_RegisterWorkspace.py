from azureml.core import Workspace
from azureml.core import Model

ws = Workspace.from_config()

model = Model.register(workspace = ws,
                       model_path ="groceryprediction.joblib",
                       model_name = "grocery_prediction",
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.23.2',
                       description = "Regression model to predict the Grocery expense price")