import React, { useState } from "react";

const BreastCancerApp = () => {
  const [formValues, setFormValues] = useState({
    radius_mean: 0,
    texture_mean: 0,
    perimeter_mean: 0,
    area_mean: 0,
    smoothness_mean: 0,
    compactness_mean: 0,
    concavity_mean: 0,
    concave_points_mean: 0,
    symmetry_mean: 0,
    fractal_dimension_mean: 0,
    radius_se: 0,
    texture_se: 0,
    perimeter_se: 0,
    area_se: 0,
    smoothness_se: 0,
    compactness_se: 0,
    concavity_se: 0,
    concave_points_se: 0,
    symmetry_se: 0,
    fractal_dimension_se: 0,
    radius_worst: 0,
    texture_worst: 0,
    perimeter_worst: 0,
    area_worst: 0,
    smoothness_worst: 0,
    compactness_worst: 0,
    concavity_worst: 0,
    concave_points_worst: 0,
    symmetry_worst: 0,
    fractal_dimension_worst: 0,
  });

  const [predictions, setPredictions] = useState([]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: value,
    });
  };

  const handlePredict = async () => {
    // Convert form values to an array
    const inputArray = Object.values(formValues).map(Number);

    try {
      // Send input data to the Flask backend for breast cancer prediction
      const response = await fetch(
        "http://localhost:5001/predict_breast_cancer",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ input_data: inputArray }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get predictions");
      }

      // Get predictions from the backend
      const data = await response.json();
      setPredictions(data.breast_cancer_prediction);
    } catch (error) {
      console.error("Error:", error.message);
    }
  };

  return (
    <div className="container breast-cancer-background">
      <div className="breast-disease-box">
        <div className="title">Breast Cancer Prediction</div>
        <form className="columns-container">
          {Object.entries(formValues).map(([name, value]) => (
            <div key={name} className="input-pair">
              <label className="label">{name}:</label>
              <input
                type="number"
                className="input-field"
                name={name}
                value={value}
                onChange={handleInputChange}
              />
            </div>
          ))}
        </form>
        <button className="button" onClick={handlePredict}>
          Predict
        </button>
        <div>
          <h3>Predictions:</h3>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>{prediction}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default BreastCancerApp;
