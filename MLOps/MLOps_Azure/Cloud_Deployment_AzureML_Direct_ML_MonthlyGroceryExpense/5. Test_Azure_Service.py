import json
import requests

input_array = [ 45000, 4 ]
scoring_uri = "URL Obtained"

# Add the 'data' field
data = { "data" : input_array, 
        "method" : "predict"} # Write it in the required format for the REST API

input_data = json.dumps(data) # Convert to JSON string

# Set the content type to JSON
headers = {"Content-Type": "application/json"}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)

# Return the model output
result = json.loads(resp.text)
# print(result)
# 'result' will contain the dictionary: {'predict': value}