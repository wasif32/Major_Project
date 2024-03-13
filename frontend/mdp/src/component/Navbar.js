// Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';
import "../style/Navbar.css"

const Navbar = () => {
    return (
        <>
         <nav>
             <input type="checkbox" id="check" />
             <label htmlFor="check" className="checkbtn">
                 <i className="fas fa-bars"></i>
             </label>
             <label className="logo">DesignX</label>
             <ul>
                 <li><Link to="/">Home</Link></li>
                 <li><Link to="/heart-disease">Heart Disease</Link></li>
                 <li><Link to="/diabetes">Diabetes</Link></li>
             <li><Link to="/breast_cancer">Breast Cancer</Link></li>
                 <li><Link to="/parkinsons">Parkinson's</Link></li>
             </ul>
         </nav>
         </>
    );
}

export default Navbar;
