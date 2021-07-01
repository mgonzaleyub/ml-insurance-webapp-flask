from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('deployment_v1')
cols = ['age','sex','bmi','children','smoker','region']


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model,data=data_unseen,round=0)
    prediction = int(prediction.Label[0])
    return render_template('index.html', pred='Expected Bill will be {}'.format(prediction))