import React, { useState } from "react";
import "./style/style.css";
import IndexPage from "./component/IndexPage";
import HeartDisease from "./component/HeartDisease";
import DiabetesDisease from "./component/DiabetesDisease";
import BreastCancerApp from "./component/BreastCancer";
import ParkinsonsApp from "./component/Parkinsons";

function Sidebar({ items, onItemClick, selectedItemIndex }) {
  return (
    <div className="sidebar">
      {items.map((item, index) => (
        <div
          key={index}
          className={`sidebar-item ${
            selectedItemIndex === index ? "selected" : ""
          }`}
          onClick={() => onItemClick(index)}
        >
          {item}
        </div>
      ))}
    </div>
  );
}

function App() {
  const [selectedItemIndex, setSelectedItemIndex] = useState(0); // Initially selected index
  const items = [
    "Home",
    "Heart Disease",
    "Diabetes",
    "Breast Cancer",
    "Parkinsons disease",
  ];

  const handleItemClick = (index) => {
    setSelectedItemIndex(index);
  };

  const renderContent = () => {
    switch (selectedItemIndex) {
      case 0:
        return <IndexPage />;
      case 1:
        return <HeartDisease />;
      case 2:
        return <DiabetesDisease />;
      case 3:
        return <BreastCancerApp />;
      case 4:
        return <ParkinsonsApp />;
      // Add cases for other items here when their content is ready
      default:
        return <div>No content available</div>;
    }
  };

  return (
    <div className="app">
      <Sidebar
        items={items}
        onItemClick={handleItemClick}
        selectedItemIndex={selectedItemIndex}
      />
      <div className="main-content">{renderContent()}</div>
    </div>
  );
}

export default App;
