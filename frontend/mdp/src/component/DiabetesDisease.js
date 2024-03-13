/*import React, { useState } from "react";

function DiabetesApp() {
  const [formValues, setFormValues] = useState({
  pregnancies: "",
  glucose: "",
  bloodPressure: "",
  skinThickness: "",
  insulin: "",
  bmi: "",
  diabetesPedigreeFunction: "",
  age: "",
});


  const [predictions, setPredictions] = useState([]);
  const [diabetesStages, setDiabetesStages] = useState([]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: parseFloat(value), // Ensure value is parsed as float
    });
  };

  const handlePredict = async () => {
    try {
      const response = await fetch("http://localhost:5001/predict_diabetes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_data: formValues }),
      });

      if (!response.ok) {
        throw new Error("Failed to get predictions");
      }

      const data = await response.json();

      // Check if the expected data is present in the response
      if (!Array.isArray(data.diabetes_prediction) || !Array.isArray(data.diabetes_stage)) {
        throw new Error("Invalid response format");
      }

      setPredictions(data.diabetes_prediction);
      setDiabetesStages(data.diabetes_stage);
    } catch (error) {
      console.error("Error:", error.message);
    }
  };

  return (
    <div className="container diabetes-background">
        <div className="diabetes-disease-box">
          <div className="title">Diabetes Prediction</div>
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
          <button className="button" onClick={handlePredict}>Predict</button>
          <div>
            <h3>Predictions:</h3>
            <ul>
              {predictions.map((prediction, index) => (
                <li key={index}>
                  Prediction: {prediction}, Stage: {diabetesStages[index]}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
  );
}

export default DiabetesApp; */


import React, { useState } from "react";

function DiabetesApp() {
  const [formValues, setFormValues] = useState({
    pregnancies: "",
    glucose: "",
    bloodPressure: "",
    skinThickness: "",
    insulin: "",
    bmi: "",
    diabetesPedigreeFunction: "",
    age: "",
  });

  const [predictions, setPredictions] = useState([]);
  const [diabetesStage, setDiabetesStage] = useState("");
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: parseFloat(value), // Ensure value is parsed as float
    });
  };

  const handlePredict = async () => {
    try {
      const response = await fetch("http://localhost:5001/predict_diabetes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_data: formValues }),
      });

      if (!response.ok) {
        throw new Error("Failed to get predictions");
      }

      const data = await response.json();

      if (!Array.isArray(data.diabetes_prediction)) {
        throw new Error("Invalid response format");
      }

      setPredictions(data.diabetes_prediction);
      if (data.diabetes_prediction[0] === 1) {
        setDiabetesStage(data.diabetes_stage);
      } else {
        setDiabetesStage(""); // Clear stage if person is non-diabetic
      }
      setError(null);
    } catch (error) {
      console.error("Error:", error.message);
      setError(error.message);
    }
  };

  return (
    <div className="container diabetes-background">
      <div className="diabetes-disease-box">
        <div className="title">Diabetes Prediction</div>
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
          {error && <div className="error-message">{error}</div>}
          <h3>Predictions:</h3>
          <ul>
            {predictions.map((prediction, index) => (
              <li key={index}>
                Person is: {prediction === 1 ? 'Diabetic' : 'Non-Diabetic'}
                <br></br>
                {prediction === 1 && `Stage: ${diabetesStage}`}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default DiabetesApp;
