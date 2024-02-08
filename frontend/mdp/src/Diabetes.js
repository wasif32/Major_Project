import React, { useState } from "react";

function DiabetesApp() {
  const [formValues, setFormValues] = useState({
    pregnancies: 0,
    glucose: 0,
    bloodPressure: 0,
    skinThickness: 0,
    insulin: 0,
    bmi: 0,
    diabetesPedigreeFunction: 0,
    age: 0,
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
      // Send input data to the Flask backend for diabetes prediction
      const response = await fetch("http://localhost:5001/predict", {
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
      setPredictions(data.prediction);
    } catch (error) {
      console.error("Error:", error.message);
    }
  };

  return (
    <div>
      <h1>Diabetes Prediction</h1>
      <form>
        {Object.entries(formValues).map(([name, value]) => (
          <div key={name}>
            <label>{name}:</label>
            <input
              type="number"
              name={name}
              value={value}
              onChange={handleInputChange}
            />
          </div>
        ))}
      </form>
      <button onClick={handlePredict}>Predict</button>
      <div>
        <h3>Predictions:</h3>
        <ul>
          {predictions.map((prediction, index) => (
            <li key={index}>{prediction}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default DiabetesApp;
