// StatsPanel.js
import React, { useEffect, useState } from "react";
import axios from "axios";
import 'bootstrap/dist/css/bootstrap.min.css';

const StatsPanel = () => {
  const [stats, setStats] = useState({
    total: 0,
    sos: 0,
    gesture: 0,
  });

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/alerts").then((res) => {
    console.log(res.data);
      const total = res.data.length;
      const sos = res.data.filter((a) => a.type === "SOS").length;
      const gesture = res.data.filter((a) => a.source === "gesture").length;
      setStats({ total, sos, gesture });
    });
  }, []);

  return (
    <div className="container my-5">
      <h2 className="text-center mb-4 text-primary">Live Alert Statistics</h2>
      <div className="row g-4">
        <div className="col-md-4">
          <div className="card border-primary shadow-sm h-100">
            <div className="card-body text-center">
              <h5 className="card-title">Total Alerts</h5>
              <p className="display-5 text-primary fw-bold">{stats.total}</p>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-danger shadow-sm h-100">
            <div className="card-body text-center">
              <h5 className="card-title">SOS Alerts</h5>
              <p className="display-5 text-danger fw-bold">{stats.sos}</p>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-warning shadow-sm h-100">
            <div className="card-body text-center">
              <h5 className="card-title">Gesture Alerts</h5>
              <p className="display-5 text-warning fw-bold">{stats.gesture}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;

