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
    '''
    Shows the index.html page.

    To return a simple html, you must use render_template and the path of html file.
    You must also add an @app.route('/route_to_go')
    :return: render_template
    '''
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    Makes prediction for the index.html insurance form
    :return: render_template
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model,data=data_unseen,round=0)
    prediction = int(prediction.Label[0])

    insurance_form_id = request.args.get('insurance_form')
    return render_template('index.html', id=insurance_form_id, pred='Expected Bill will be {}'.format(prediction))