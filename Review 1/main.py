# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='.', template_folder='.')
model = load_model('brnn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['frame.protocols']).reshape(1, -1)
    prediction = model.predict(input_data)
    result = prediction.tolist()
    return jsonify({'result': result})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)