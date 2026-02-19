from flask import Flask, render_template, request
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Get current folder path (important for VS Code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model files safely
model = pickle.load(open(os.path.join(BASE_DIR, 'crop_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)
        output = label_encoder.inverse_transform(prediction)

        return render_template('index.html',
                               prediction_text="Recommended Crop: {}".format(output[0]))
    except Exception as e:
        return render_template('index.html',
                               prediction_text="Error: Please enter valid numeric values!")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
