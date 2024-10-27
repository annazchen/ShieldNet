import os
from flask import Flask, request, render_template, url_for
from ..w.Graph import analyze_input_file

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
        "data": png_path[1],  # Binary predictions or processed data output
        "img_path": png_path[0]  # Image path
    })
    
    # Redirect to display the results
    return redirect('/display')

@app.route('/display', methods=["GET"])
def display():
    if predictions:
        last_prediction = predictions[-1]
        return render_template(
            'main.html',
            prediction_text=last_prediction["data"],
            img_url=url_for('static', filename=last_prediction["img_path"])
        )
    return render_template('main.html', prediction_text="No prediction available")

if __name__ == "__main__":
    app.run()
