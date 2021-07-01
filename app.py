from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np

app = Flask(__name__)

#---- CONFIG -------------
# Name for the model to use
model_name = 'deployment_v1'

# Columns existing in the model
cols = ['age','sex','bmi','children','smoker','region']
# ------------------------

model = load_model('trained_models/'+model_name)

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