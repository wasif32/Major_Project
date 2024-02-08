from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained diabetes model
loaded_diabetes_model = joblib.load('diabetes_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'input_data' not in data:
            raise ValueError("Input data not provided")

        input_data = np.array(data['input_data']).reshape(1, -1)

        # Make predictions using the loaded diabetes model
        prediction = loaded_diabetes_model.predict(input_data)

        # Print the prediction in the backend terminal
        print("Prediction:", prediction)

        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
