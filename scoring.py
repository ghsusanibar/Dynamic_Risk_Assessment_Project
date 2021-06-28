from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score
#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])   #practicemodels
test_data_path = os.path.join(config['test_data_path'])         #testdata
model_path = os.path.join(config['output_model_path'])          #practicemodels
#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    testdata=pd.read_csv(os.getcwd() + '/' + test_data_path + '/' + 'testdata.csv')
    X=testdata.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y=testdata['exited'].values.reshape(-1, 1).ravel()
    with open(os.getcwd() + '/' + model_path + '/' + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    predicted=model.predict(X)
    f1score=metrics.f1_score(predicted,y)
    with open(os.getcwd() + '/' + model_path + '/latestscore.txt', 'w') as f:
        f.write(str(f1score))

if __name__ == '__main__':
    score_model()