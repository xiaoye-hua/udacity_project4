#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    app_test.py
# @Author:      Hua Guo
# @Time:        28/08/2022 09:29
# @Desc:


from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

def readpandas(filename):
    thedata=pd.read_csv(filename)
    return thedata

@app.route('/prediction')
def prediction():
    # thedata=pd.read_csv('predictiondata.csv')
    # with open('deployedmodel.pkl', 'rb') as file:
    #     themodel = pickle.load(file)
    # prediction=themodel.predict(thedata)
    return str("hah")

app.run(host='0.0.0.0', port=4000, debug=True)