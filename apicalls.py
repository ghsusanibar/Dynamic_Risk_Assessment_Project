import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

#Call each API endpoint and store the responses
response1 = requests.get('http://127.0.0.1:8000/prediction?filename=./testdata/testdata.csv').content
response2 = requests.get('http://127.0.0.1:8000/scoring').content
response3 = requests.get('http://127.0.0.1:8000/summarystats').content
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content

#combine all API responses
responses = response1 + '\n' + response2 + '\n' + response3 + '\n' + response4 + '\n'

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 
model_path = os.path.join(config['output_model_path'])
with open(os.getcwd() + '/' + model_path + '/apireturns.txt', 'w') as f:
    f.write(str(responses))