import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path'])   #production_deployment
test_data_path = os.path.join(config['test_data_path'])               #testdata
dataset_csv_path = os.path.join(config['output_folder_path'])         #ingesteddata
##################Function to get model predictions
def model_predictions(df):
    #read the deployed model and a test dataset, calculate predictions
#     testdata=pd.read_csv(os.getcwd() + '/' + test_data_path + '/' + 'testdata.csv')
    X=df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y=df['exited'].values.reshape(-1, 1).ravel()
    with open(os.getcwd() + '/' + prod_deployment_path + '/' + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    predicted=model.predict(X)
    return list(predicted)
    #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    df=pd.read_csv(os.getcwd() + '/' + dataset_csv_path + '/' + 'finaldata.csv')
    #calculate summary statistics here
    themeans=list(df[['lastmonth_activity','lastyear_activity','number_of_employees']].mean())
    themedians=list(df[['lastmonth_activity','lastyear_activity','number_of_employees']].median())
    thestanddesv=list(df[['lastmonth_activity','lastyear_activity','number_of_employees']].std())
    return [themeans, themedians, thestanddesv]
    #return value should be a list containing all summary statistics

##################Function to get timings
def dataframe_missing():
    df=pd.read_csv(os.getcwd() + '/' + dataset_csv_path + '/' + 'finaldata.csv')
    return list(df.isna().sum()/df.shape[0])

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ingestion=timeit.default_timer() - starttime
    
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing_trining=timeit.default_timer() - starttime
    
    return [timing_ingestion, timing_trining]
    #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    return outdated


if __name__ == '__main__':
    testdata=pd.read_csv(os.getcwd() + '/' + test_data_path + '/' + 'testdata.csv')
    print(model_predictions(testdata))
    print(dataframe_summary())
    print(dataframe_missing())
    print(execution_time())
    print(outdated_packages_list())