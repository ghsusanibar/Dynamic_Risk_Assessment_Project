import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix
from pylab import savefig
from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])          #practicemodels
test_data_path = os.path.join(config['test_data_path'])               #testdata
##############Function for reporting
def score_model():
#     df=pd.read_csv(os.getcwd() + '/' + test_data_path + '/' + 'testdata.csv')
#     print(model_predictions(df))
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    testdata=pd.read_csv(os.getcwd() + '/' + test_data_path + '/' + 'testdata.csv')
    y=testdata['exited'].values.reshape(-1, 1).ravel()
    y_pred = model_predictions(testdata)
    cm=confusion_matrix(y,y_pred)
    svm = sns.heatmap(cm, annot=True)
    figure = svm.get_figure()    
    figure.savefig(os.getcwd() + '/' + model_path + '/' + 'confusionmatrix.png', dpi=400)

if __name__ == '__main__':
    score_model()