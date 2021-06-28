from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, dataframe_missing, outdated_packages_list
from scoring import score_model
######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prediction_model = None

def readpandas(filename):
    thedata=pd.read_csv(filename)
    return thedata
#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    df=readpandas(filename)
    y_pred = model_predictions(df)
    return str(y_pred)
    #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1score = score_model()
    return str(f1score)
    #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary = dataframe_summary()
    return str(summary)
    #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnost():        
    #check timing and percent NA values
    missing = dataframe_missing()
    timing = execution_time()
    outdated = outdated_packages_list()
    return str(missing) + '\n' + str(timing) + '\n' + str(outdated) + '\n'
    #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)