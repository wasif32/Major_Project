import React, { useState } from "react";

function App() {
  const [formValues, setFormValues] = useState({
    age: 0,
    sex: 0,
    cp: 0,
    trestbps: 0,
    chol: 0,
    fbs: 0,
    restecg: 0,
    thalach: 0,
    exang: 0,
    oldpeak: 0,
    slope: 0,
    ca: 0,
    thal: 0,
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
      // Send input data to the Flask backend
      const response = await fetch("http://localhost:5000/predict", {
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

export default App;
