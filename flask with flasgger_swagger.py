# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:24:10 2022

@author: 91709
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('rfclassifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict')
def predict_note_auth():
    """ BANK NOTES AUTHENTICATION
    This is using DOCSTRING for specifications.
    ----
    parameters:
        - name : variance
          in: query
          type : number
          required : true
        - name : skewness
          in: query
          type : number
          required : true
        - name : kurtosis
          in: query
          type : number
          required : true
        - name : entropy
          in: query
          type : number
          required : true
    responses:
        200:
            description: The output values 
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy  = request.args.get('entropy')
    pred = classifier.predict([[variance, skewness, kurtosis, entropy]])
    return f"predicted value is {str(pred)}"


@app.route('/predict_file', methods=['POST'])
def predict_note_auth_file():
    """ BANK NOTES AUTHENTICATION
    This is using DOCSTRING for Specification.
    ---
    parameters:
        - name : file
          in : formData
          type: file
          required : true
    responses:
        200:
            description: the output values
    
    """
    df_test = pd.read_csv(request.files.get('file'))
    pred=classifier.predict(df_test)
    
    return f"predicted value is {str(list(pred))}"

    


if __name__ =='__main__':
    app.run()
