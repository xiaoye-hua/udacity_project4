from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, missing_data, outdated_packages_list



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST']
           )
def predict():        
    #call the prediction function you created in Step 3
    path = request.args.get('path')
    # print(path)
    # path = os.path.join(dataset_csv_path, 'finaldata.csv')
    res = model_predictions(path)
    return str(res)

# #######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    score = score_model()
    return str(score) #add return value (a single F1 score number)
#
# #######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['POST','OPTIONS'])
def stats():
    path = request.args.get('path')
    df = pd.read_csv(path)
    res = dataframe_summary(df=df)
    #check means, medians, and modes for each column
    return str(res) #return a list of all calculated summary statistics
#
# #######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['POST','OPTIONS'])
def diagnostics():
    path = request.args.get('path')
    df = pd.read_csv(path)
    res1 = missing_data(df=df)
    res2 = execution_time()
    res3 = outdated_packages_list()

    return {
        'missing_data': res1,
        'execution_time': res2,
        # 'outdated_packages': res3

    }#add return value for all diagnostics


if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=9000, debug=True
            # , threaded=True
            )
