
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import subprocess as sp
import time
import requests

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(path):
    df = pd.read_csv(path)
    #read the deployed model and a test dataset, calculate predictions
    pipeline = joblib.load(os.path.join(prod_deployment_path, 'trainedmodel.pkl'))
    y_pred = pipeline.predict(df)
    assert len(y_pred) == df.shape[0]
    return y_pred

##################Function to get summary statistics
def dataframe_summary(df):
    x = df.drop('corporation', axis=1)
    res = {}
    for col in x.columns:
        res[col] = [
            np.mean(x[col].values)
            , np.median(x[col].values)
            , np.std(x[col].values)
        ]
    print(res)
    return res #return value should be a list containing all summary statistics


def missing_data(df):
    res = df.isna().sum()/len(df)
    res = res.reset_index()[0].values.tolist()
    print(res)
    return res

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    cmd_lst = [
        'python ingestion.py',
        'python training.py'
    ]
    res = []
    for cmd in cmd_lst:
        begin = time.time()
        sp.run(cmd, shell=True)
        end = time.time()
        duration = end - begin
        res.append(duration)
    print(res)
    return res #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of

    with open('requirements.txt', 'r') as f:
        lst = f.readlines()

    package_lst = []
    current_lst = []
    latest_lst = []
    for row in lst:
        package, current_version,  = row.replace('\n', '').split('==')
        response = requests.get(f'https://pypi.org/pypi/{package}/json')
        latest_version = response.json()['info']['version']
        package_lst.append(package)
        current_lst.append(current_version)
        latest_lst.append(latest_version)
    df = pd.DataFrame(
        {
            'package': package_lst,
            'current_version': current_lst,
            'latest_version': latest_lst
        }
    )
    print(df)
    return df


if __name__ == '__main__':
    path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(path)

    model_predictions(path=path)
    dataframe_summary(df=df)

    missing_data(df=df)
    execution_time()
    outdated_packages_list()





    
