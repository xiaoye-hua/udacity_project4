from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import json
import joblib

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y = df['exited']
    X = df.drop('exited', axis=1)
    #use this logistic regression for training
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    data_process = ColumnTransformer(
        transformers=[
            ('onehot', onehot_encoder, ['corporation']),
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(
        [
            ('data_process', data_process)
        , ('model', model)
        ]
    )
    
    #fit the logistic regression to your data
    pipeline.fit(X=X, y=y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    joblib.dump(
        value=pipeline,
        filename=os.path.join(model_path, 'trainedmodel.pkl')
    )


if __name__ == "__main__":
    train_model()