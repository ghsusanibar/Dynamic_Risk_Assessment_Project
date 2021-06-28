# Dynamic_Risk_Assessment_Project

## Overview
This project is part of the Udacity ML Devops Engineer nanodegree program.

## Summary
The goal of the project is to implement a dynamic risk assessment model. The project has five steps in order to ingest the data, traning and scoring a model, make diagnostics and report trhough an API and finally automate the re-training process.

The initial values from config.json are:
- input_folder_path: "practicedata"
- output_folder_path: "ingesteddata"
- test_data_path: "testdata"
- output_model_path: "practicemodels"
- prod_deployment_path: "production_deployment"

## Step 1: Data Ingestion
First I checked if there are datasets the **input_folder_path** and then I combined all the files in one dataset. I deduplicated the daframe and then I saved the result in the **output_folder_path** with the name *finaldata.csv* and I also generated the *ingestedfiles.txt* file containing the records of the datasets used in the ingestion.

## Step 2: Training, Scoring, and Deploying an ML Model
- Training
First I read the *finaldata.csv* dataset from the previous step, then I trained a logistic regression model and finally I saved the model in the **output_model_path** with the name *trainedmodel.pkl*.
- Scoring
Again I read the *finaldata.csv* dataset from the previous step, then I read the *trainedmodel.pkl* model, I got the predictions using the trained model in order to calculate the F1 metric score. Finall I saved the metric in the **output_model_path** with the name *latestscore.txt*.
- Deploying
For this step I just copied the *trainedmodel.pkl*, *latestscore.txt* and *ingestedfiles.txt* files from their original directory to the **prod_deployment_path**.

## Step 3: Diagnostics
This step is about write five functions to get information from the trained model and the dataset.
- model_predictions()
This function reads the deployed model and a test dataset. Then it calculates and returns predictions from the model.
- dataframe_summary()
This function reads the test dataset and then it calculates and returns the mean, median and standard deviation from the numeric columns of the test dataset.
- dataframe_missing()
This fucntion reads the test dataset and then it calculates the percentage of missing values from each column of the test dataset.
- execution_time()
This fucntion calculates the execution time of the *ingestion.py* and *training.py* scripts.
- outdated_packages_list()
This function lists the pip packages outdated using the following command: pip list --outdated

## Step 4: Reporting
- Reporting
The *reporting.py* script reads the test dataset, it uses the *model_model_predictions* function from the previous step to calculate the predictions, then it generates the confusopn matrix and finnaly it saves the plotting image of the confusion matrix.

- API setup
In the *app.py* script the API is created with four endpoints. 
1) Prediction Endpoint (/prediction)
This endpoint reads the dataset using the filename provided and then it uses the *model_model_predictions* function to calculate and return the predictions.
2) Scoring Endpoint (/scoring)
This endpoint uses the *score_model* function to score the trained model.
3) Summary Statistics Endpoint (/summarystats)
This endpoint uses the *dataframe_summary* function to get the means, medians and standard deviation from the numeric columns of the test dataset.
4) Diagnostics Endpoint (/diagnostics)
This endpoint checks the execution time of the funcions, gets the missing percentage of the dataset columns and gets the outdated packages from pip. Finally it returns all this diagnostics information.
5) Calling API endpoints
The four endpoints from the API are called in the *apicalls.py* script. The *requests* module is used to call the endpoints and get the returned information.

## Step 5: Process Automation
First this step reads the *ingestedfiles.txt* file to get the list of the datasets used to train the deployed model. As the config file is edited, now the **input_folder_path** is the *sourcedata* directory. As this new directory has new datasets, so the script has to check if these datasets are listed in the *ingestedfiles.txt* file. If the new datsets are not listed in the *ingestedfiles.txt* file so the script executes the *ingestion.py* file to ingest this new data to the final dataset. Then the model is retrained and scored. The next step is to check model drift. If we find model drift we have to execute the *training.py* file to retrained the model. In my case, I did not found model drift as the F1 score from step 3 is not lower than the laltes F1 score of the deployed model. 
Finally I wrote a crontab file that runs the fullprocess.py script one time every 10 min. The command is "10 * * * * python /home/workspace/fullprocess.py".