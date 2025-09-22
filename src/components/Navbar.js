// Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4">
      <Link className="navbar-brand fw-bold" to="/">SAFENET</Link>
      <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
        <span className="navbar-toggler-icon"></span>
      </button>
      <div className="collapse navbar-collapse" id="navbarNav">
        <ul className="navbar-nav ms-auto">
          <li className="nav-item">
            <Link className="nav-link" to="/">Home</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/live">Live Alerts</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/alerts">Alert History</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/map">Map</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/stats">Stats</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;

