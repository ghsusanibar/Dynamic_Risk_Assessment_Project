import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import ast
import subprocess
import sys

import training
import scoring
import deployment
import diagnostics
import reporting

import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

prod_path = config['prod_deployment_path']
source_path = config['input_folder_path']
ingested_path = config['output_folder_path']
##################Check and read new data
#first, read ingestedfiles.txt
with open(os.getcwd() + '/' + prod_path + '/ingestedfiles.txt', 'r') as f:
    datalist = ast.literal_eval(f.read())
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(os.getcwd() + '/' + source_path)
list_files = []
for each_filename in filenames:
    if each_filename not in datalist:
        list_files.append(each_filename)
if len(list_files) != 0:
    print('Re-Ingestion...')
    subprocess.call(['python', 'ingestion.py'])
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(list_files) == 0:
    sys.exit()
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.getcwd() + '/' + prod_path + '/latestscore.txt', 'r') as f:
    latestscore = float(f.read())
print(latestscore)
with open(os.getcwd() + '/' + prod_path + '/' + 'trainedmodel.pkl', 'rb') as file:
    model = pickle.load(file)
df=pd.read_csv(os.getcwd() + '/' + ingested_path + '/' + 'finaldata.csv')
X=df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
y=df['exited'].values.reshape(-1, 1).ravel()
predicts = model.predict(X)
f1score=metrics.f1_score(predicts, y)
print(f1score)
modeldrift_state = False
if(f1score < latestscore):
    modeldrift_state = True
    print('Raw model drift has ocurred')
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if modeldrift_state == False:
    print('Model drift not found...')
    sys.exit()
else:
    print('Retraining...')
    subprocess.call(['python', 'training.py'])

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if modeldrift_state == True:
    print('Re-deployment...')
    subprocess.call(['python', 'deployment.py'])
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
print('Running diagnostics, reporting and appicalls scripts...')
subprocess.call(['python', 'diagnostics.py'])
subprocess.call(['python', 'reporting.py'])
subprocess.call(['python', 'appicalls.py'])
print('End fullprocess execution!')