# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from aco_model import ACO  # Import the ACO class from the separate module

app = Flask(__name__)
CORS(app)

# Load the trained ACO model
loaded_aco_model = joblib.load('aco2_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'input_data' not in data:
            raise ValueError("Input data not provided")

        input_data = np.array(data['input_data']).reshape(1, -1)

        # Get the selected features from the loaded ACO model
        selected_features = loaded_aco_model.accuracies[-1].obtainSolution_final()

        # Use the selected features to extract relevant columns from the input data for prediction
        input_data_selected = input_data[:, selected_features]

        # Make predictions
        prediction = loaded_aco_model.clf.predict(input_data_selected)

        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
