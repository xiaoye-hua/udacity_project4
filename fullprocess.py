

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import sys
import pandas as pd
import logging
import joblib
import subprocess as sp
from sklearn import metrics

logging.basicConfig(level=logging.DEBUG)

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

##################Check and read new data
#first, read ingestedfiles.txt

with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    lst = f.readline().replace('\n', '').split(',')
logging.debug(lst)
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

file_list = [os.path.join(input_folder_path, name) for name in os.listdir(input_folder_path) if '.csv' in name]

new_file = False
logging.debug(file_list)
for name in file_list:
    if name not in lst:
        new_file = True
        break
logging.debug(new_file)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_file:
    sys.exit("There isn't new data -> exit program")

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
    original_f1 = float(f.readline().replace('\n', ''))
df_lst = [pd.read_csv(name) for name in file_list]
df = pd.concat(df_lst)
pipeline = joblib.load(os.path.join(model_path, 'trainedmodel.pkl'))
y = df['exited']
X = df.drop('exited', axis=1)
y_pred = pipeline.predict(X)
new_f1 = metrics.f1_score(y_true=y, y_pred=y_pred)
logging.debug(f"original_f1: {original_f1}; new_f1: {new_f1}")

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 >= original_f1:
    sys.exit("F1 doesn't decrease -> exit program")

####################  re-process data; re-run model training
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

cmd_lst = [
    'python ingestion.py',
    'python training.py',
    'python deployment.py',
    'python diagnostics.py',
    'python reporting.py'
]
for cmd in cmd_lst:
    res = sp.run(cmd, shell=True)
    logging.debug(f"{cmd}: {res}")








