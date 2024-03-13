# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)
# CORS(app)

# # Load the trained diabetes model
# loaded_diabetes_model = joblib.load('diabetes_model.joblib')

# # Load the trained Parkinson's disease model
# loaded_parkinsons_model = joblib.load('parkinsons.joblib')

# # Load the trained breast cancer detection ANN model
# loaded_breast_cancer_model = load_model('ann_model.h5')

# # Load the scaler used for breast cancer detection model
# scaler = joblib.load('scaler.pkl')

# @app.route('/predict_diabetes', methods=['POST'])
# def predict_diabetes():
#     try:
#         data = request.get_json()

#         if 'input_data' not in data:
#             raise ValueError("Input data not provided")

#         input_data = np.array(data['input_data']).reshape(1, -1)

#         # Make predictions using the loaded diabetes model
#         prediction = loaded_diabetes_model.predict(input_data)

#         # Print the prediction in the backend terminal
#         print("Diabetes Prediction:", prediction)

#         return jsonify({'diabetes_prediction': prediction.tolist()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/predict_parkinsons', methods=['POST'])
# def predict_parkinsons():
#     try:
#         data = request.get_json()

#         if 'input_data' not in data:
#             raise ValueError("Input data not provided")

#         input_data = np.array(data['input_data']).reshape(1, -1)

#         # Make predictions using the loaded Parkinson's disease model
#         prediction = loaded_parkinsons_model.predict(input_data)

#         # Print the prediction in the backend terminal
#         print("Parkinson's Prediction:", prediction)

#         return jsonify({'parkinsons_prediction': prediction.tolist()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/predict_breast_cancer', methods=['POST'])
# def predict_breast_cancer():
#     try:
#         data = request.get_json()

#         if 'input_data' not in data:
#             raise ValueError("Input data not provided")

#         input_data = np.array(data['input_data']).reshape(1, -1)

#         # Scale the input features
#         input_data_scaled = scaler.transform(input_data)

#         # Make predictions using the loaded breast cancer detection model
#         prediction = loaded_breast_cancer_model.predict(input_data_scaled)

#         # Print the prediction in the backend terminal
#         print("Breast Cancer Prediction:", prediction)

#         return jsonify({'breast_cancer_prediction': prediction.tolist()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load the trained diabetes model
loaded_diabetes_model = joblib.load('diabetes_model.joblib')

# Load the trained Parkinson's disease model
loaded_parkinsons_model = joblib.load('parkinsons.joblib')

# Load the trained breast cancer detection ANN model
loaded_breast_cancer_model = load_model('ann_model.h5')

# Load the scaler used for breast cancer detection model
scaler = joblib.load('scaler.pkl')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()

        if 'input_data' not in data:
            raise ValueError("Input data not provided")

        # Extract numerical values from nested dictionary
        input_data_dict = data['input_data']
        input_data = [input_data_dict[key] for key in input_data_dict]

        # Convert to numpy array and reshape
        input_data = np.array(input_data).reshape(1, -1)

        # Make predictions using the loaded diabetes model
        prediction = loaded_diabetes_model.predict(input_data)

        # Map predictions to diabetes stages
        stages = ["Pre-diabetes", "Type 2 diabetes", "Type 1 diabetes", "Gestational diabetes"]
        stage_prediction = stages[np.argmax(prediction)]

        return jsonify({'diabetes_prediction': prediction.tolist(), 'diabetes_stage': stage_prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        data = request.get_json()

        if 'input_data' not in data:
            raise ValueError("Input data not provided")

        input_data = np.array(data['input_data']).reshape(1, -1)

        # Print the input data in the backend terminal
        print("Input data for Parkinson's prediction:", input_data)

        # Make predictions using the loaded Parkinson's disease model
        prediction = loaded_parkinsons_model.predict(input_data)

        # Print the prediction in the backend terminal
        print("Parkinson's Prediction:", prediction)

        return jsonify({'parkinsons_prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_breast_cancer', methods=['POST'])
def predict_breast_cancer():
    try:
        data = request.get_json()

        if 'input_data' not in data:
            raise ValueError("Input data not provided")

        input_data = np.array(data['input_data']).reshape(1, -1)

        # Print the input data in the backend terminal
        print("Input data for breast cancer prediction:", input_data)

        # Scale the input features
        input_data_scaled = scaler.transform(input_data)

        # Make predictions using the loaded breast cancer detection model
        prediction = loaded_breast_cancer_model.predict(input_data_scaled)

        # Print the prediction in the backend terminal
        print("Breast Cancer Prediction:", prediction)

        return jsonify({'breast_cancer_prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
