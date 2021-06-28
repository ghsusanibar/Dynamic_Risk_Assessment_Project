import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_combine = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    filenames = os.listdir(os.getcwd() + '/' + input_folder_path)
    list_files = []
    for each_filename in filenames:
        df1 = pd.read_csv(os.getcwd() + '/' + input_folder_path + '/' + each_filename)
        df_combine=df_combine.append(df1,ignore_index=True)
        list_files.append(each_filename)
    df_combine = df_combine.drop_duplicates()
    if not os.path.exists(os.getcwd() + '/' + output_folder_path):
        os.makedirs(os.getcwd() + '/' + output_folder_path)
    df_combine.to_csv(os.getcwd() + '/' + output_folder_path + '/finaldata.csv', index=False)
    with open(os.getcwd() + '/' + output_folder_path + '/ingestedfiles.txt', 'w') as f:
        f.write(str(list_files))

if __name__ == '__main__':
    merge_multiple_dataframe()