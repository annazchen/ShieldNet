import numpy as np
import os
from flask import Flask, redirect, request, render_template
import pickle
from ..w.Graph import analyze_input_file
app = Flask(__name__, static_url_path='/static')

# def create_app(test_config=None):
#     # create and configure the app
#     app = Flask(__name__, instance_relative_config=True)
#     app.config.from_mapping(
#         SECRET_KEY='dev',
#         DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
#     )

#     if test_config is None:
#         # load the instance config, if it exists, when not testing
#         app.config.from_pyfile('config.py', silent=True)
#     else:
#         # load the test config if passed in
#         app.config.from_mapping(test_config)

#     # ensure the instance folder exists
#     try:
#         os.makedirs(app.instance_path)
#     except OSError:
#         pass

def test():
    return "Please work"

# model = pickle.load(open('skaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('main.html')
predictions = []
@app.route('/predict', methods=["POST"])
def predict():
    csv_file = request.files['csv_file']
    prediction = analyze_input_file(csv_file)
    predictions.append(prediction)
    return prediction
@app.route('/display', methods=["GET"])
def hello():
    return render_template('main.html', prediction_text=predictions[-1])
app.run()
    # return app