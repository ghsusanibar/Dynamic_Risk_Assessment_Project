from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copyfile
##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])                 #practicemodels
prod_deployment_path = os.path.join(config['prod_deployment_path'])     #production_deployment
ingest_path = config['output_folder_path']                              #ingesteddata
####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    copyfile(os.getcwd() + '/' + model_path + '/' + 'trainedmodel.pkl', os.getcwd() + '/' + prod_deployment_path + '/' + 'trainedmodel.pkl')
    copyfile(os.getcwd() + '/' + model_path + '/' + 'latestscore.txt', os.getcwd() + '/' + prod_deployment_path + '/' + 'latestscore.txt')
    copyfile(os.getcwd() + '/' + ingest_path + '/' + 'ingestedfiles.txt', os.getcwd() + '/' + prod_deployment_path + '/' + 'ingestedfiles.txt')
    
if __name__ == '__main__':
    store_model_into_pickle()