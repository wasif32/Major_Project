// import React from 'react';
// import { Link } from 'react-router-dom';

// const IndexPage = () => {
//   return (
//     <>
      
//       <div>
//       <h1>Disease Detection</h1>
//       <div className="disease-cards">
//         <div className="disease-card">
//           <h2>Heart Disease</h2>
//           <p>Description of heart disease</p>
//           <Link to="/heart-disease">
//             <button>View Heart Disease</button>
//           </Link>
//         </div>
//         <div className="disease-card">
//           <h2>Diabetes</h2>
//           <p>Description of diabetes</p>
//           <Link to="/diabetes">
//             <button>View Diabetes</button>
//           </Link>
//         </div>
//         <div className="disease-card">
//           <h2>Breast Cancer</h2>
//           <p>Description of BreastCancer</p>
//           <Link to="/breast_cancer">
//             <button>View BreastCancer</button>
//           </Link>
//         </div>
//         <div className="disease-card">
//           <h2>ParkinsonsApp</h2>
//           <p>Description of Parkinsons</p>
//           <Link to="/parkinsons">
//             <button>View Parkinsons</button>
//           </Link>
//         </div>
    
//       </div>
//     </div>
//     </>
//   )
// }

// export default IndexPage




import React from 'react';

const IndexPage = () => {
  return (
    <>
    <div className="home-content">
      <h1>Multiple Disease Detection</h1>
      <p>Major project for year 2023-2024</p>
      <p>Project by:</p>
      <ul>
        <li>A06 Sachin Kumavat</li>
        <li>A19 Wasif Khan</li>
        <li>A38 Sakshi Kale</li>
        <li>A52 Ravi Vishwakarma</li>
      </ul>

      <div className="prediction-section">
        <div className="prediction-item">
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMwVVW27FTWjHJenZaE51wdxMHNjsxgr1Qag&usqp=CAU" alt="Heart Disease Prediction" />
          <h3>Heart Disease Prediction</h3>
          <ul>
            <li>Algoritm </li>
            <li>Accuracy</li>
            <li>Small content point 3</li>
          </ul>
        </div>
        <div className="prediction-item">
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-2cQCEiIBJ3i38WMsmC2l6LUNbqxFZGsUo8aWj1_15cHPZmkSMsLQI3ZTARLP7Qnh3H8&usqp=CAU" alt="Diabetes Prediction" />
          <h3>Diabetes Prediction</h3>
          <ul>
            <li>Algoritm </li>
            <li>Accuracy</li>
            <li>Small content point 3</li>
          </ul>
        </div>
        <div className="prediction-item">
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYrUHfmmk38e2sbAcFWYqdO1aurM4HDECWpA&usqp=CAU" alt="Breast Cancer Prediction" />
          <h3>Breast Cancer Prediction</h3>
          <ul>
            <li>Algoritm </li>
            <li>Accuracy</li>
            <li>Small content point 3</li>
          </ul>
        </div>
        <div className="prediction-item">
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrbkW4NqZj92Lmte8Yy4tFqsEYbuykGgyeJg&usqp=CAU" alt="Parkinson's Disease Prediction" />
          <h3>Parkinson's Disease Prediction</h3>
          <ul>
            <li>Algoritm </li>
            <li>Accuracy</li>
            <li>Small content point 3</li>
          </ul>
        </div>
      </div>
    </div>
    </>
  )
}

export default IndexPage

