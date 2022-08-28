import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
# test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    file_list = [os.path.join(dataset_csv_path, name) for name in os.listdir(dataset_csv_path) if '.csv' in name]
    df_lst = [pd.read_csv(name) for name in file_list]
    df = pd.concat(df_lst)
    y = df['exited']
    X = df.drop('exited', axis=1)
    pipeline = joblib.load(os.path.join(prod_deployment_path, 'trainedmodel.pkl'))
    plot_confusion_matrix(estimator=pipeline, X=X, y_true=y)
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()
