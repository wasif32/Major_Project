import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import ParkinsonsApp from "./Parkinsons";
import reportWebVitals from "./reportWebVitals";

ReactDOM.render(
  <React.StrictMode>
    <ParkinsonsApp />
  </React.StrictMode>,
  document.getElementById("root")
);

reportWebVitals();
