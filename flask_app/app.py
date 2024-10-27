import os
from flask import Flask, request, render_template, url_for, redirect
from Graph import analyze_input_file

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('main.html')

predictions = []

@app.route('/predict', methods=["POST"])
def predict():
    csv_file = request.files['csv_file']
    
    # Analyze file and get prediction along with the path to the PNG file
    png_path = analyze_input_file(csv_file)  # Returns both predictions and PNG path
    
    # Store the latest prediction details
    predictions.append({
        # "data": png_path[1],  # Binary predictions or processed data output
        png_path  # Image path
    })
    
    # Redirect to display the results
    return redirect('/display')

@app.route('/display', methods=["GET"])
def display():
    if predictions:
        last_prediction = predictions[-1]
        return render_template(
            'main.html',
            prediction_text="../"+last_prediction
            # img_url="../"+last_prediction
        )
    # return render_template('main.html', prediction_text="Prediction available", img_url = img_url_temp)

if __name__ == "__main__":
    app.run()
