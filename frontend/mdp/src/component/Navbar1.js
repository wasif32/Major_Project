import React, { useState } from "react";
import "../style/Navbar1.css";
import { Link, NavLink } from "react-router-dom";

const Navbar1 = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav>
      <Link to="/" className="title">
        Website
      </Link>
      <div className="menu" onClick={() => setMenuOpen(!menuOpen)}>
        <span></span>
        <span></span>
        <span></span>
      </div>
      <ul className={menuOpen ? "open" : ""}>
        <li>
          <NavLink to="/heart-disease">HeartDisease</NavLink>
        </li>
        <li>
          <NavLink to="/diabetes">Diabetes</NavLink>
        </li>
        <li>
          <NavLink to="/breast_cancer">Breast Cancer</NavLink>
        </li>
        <li>
          <NavLink to="/parkinsons">Parkinsons</NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar1;