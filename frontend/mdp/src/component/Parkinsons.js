import React, { useState } from "react";

const ParkinsonsApp = () => {
  const [formValues, setFormValues] = useState({
    "MDVP:Fo(Hz)": 0,
    "MDVP:Fhi(Hz)": 0,
    "MDVP:Flo(Hz)": 0,
    "MDVP:Jitter(%)": 0,
    "MDVP:Jitter(Abs)": 0,
    "MDVP:RAP": 0,
    "MDVP:PPQ": 0,
    "Jitter:DDP": 0,
    "MDVP:Shimmer": 0,
    "MDVP:Shimmer(dB)": 0,
    "Shimmer:APQ3": 0,
    "Shimmer:APQ5": 0,
    "MDVP:APQ": 0,
    "Shimmer:DDA": 0,
    NHR: 0,
    HNR: 0,
    RPDE: 0,
    DFA: 0,
    spread1: 0,
    spread2: 0,
    D2: 0,
    PPE: 0,
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
      // Send input data to the Flask backend for Parkinson's prediction
      const response = await fetch("http://localhost:5001/predict_parkinsons", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_data: inputArray }),
      });

      if (!response.ok) {
        throw new Error("Failed to get predictions");
      }

      // Get predictions from the backend
      const data = await response.json();
      setPredictions(data.parkinsons_prediction);
    } catch (error) {
      console.error("Error:", error.message);
    }
  };

  return (
    <div className="container parkinsons-disease-background">
        <div className="parkinsons-disease-box">
          <div className="parkinsons-title">Parkinson's Disease Prediction</div>
          <form className="parkinsons-disease-columns-container">
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
          <button className="button" onClick={handlePredict}>Predict</button>
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
}

export default ParkinsonsApp;
