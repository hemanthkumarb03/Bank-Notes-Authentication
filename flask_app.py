# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 08:58:45 2022

@author: 91709
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

pickle_in = open('rfclassifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict')
def predict_note_auth():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy  = request.args.get('entropy')
    pred = classifier.predict([[variance, skewness, kurtosis, entropy]])
    return f"predicted value is {str(pred)}"


@app.route('/predict_file', methods=['POST'])
def predict_note_auth_file():
    df_test = pd.read_csv(request.files.get('file'))
    pred=classifier.predict(df_test)
    
    return f"predicted value is {str(list(pred))}"

    


if __name__ =='__main__':
    app.run()
