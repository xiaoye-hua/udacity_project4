from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import joblib



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    pipeline = joblib.load(os.path.join(model_path, 'trainedmodel.pkl'))

    file_list = [os.path.join(test_data_path, name) for name in os.listdir(test_data_path) if '.csv' in name]
    df_lst = [pd.read_csv(name) for name in file_list]
    df = pd.concat(df_lst)
    y = df['exited']
    X = df.drop('exited', axis=1)
    y_pred = pipeline.predict(X)
    f1 = metrics.f1_score(y_true=y, y_pred=y_pred)
    file_path = os.path.join(model_path, 'latestscore.txt')
    # print(file_path)
    with open(file_path, 'w') as f:
        f.writelines(str(f1))
    return f1


if __name__ == "__main__":
    score_model()
