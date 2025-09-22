// AlertHistoryTable.js
import React, { useEffect, useState } from "react";
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const AlertHistoryTable = () => {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/alerts')
      .then(res => setAlerts(res.data))
      .catch(err => console.error('Error fetching alerts:', err));
  }, []);

  return (
    <div className="container my-5">
      <h2 className="text-center mb-4 text-success">Alert History</h2>
      <div className="table-responsive">
        <table className="table table-bordered table-hover">
          <thead className="table-dark">
            <tr>
              <th>Type</th>
              <th>Location</th>
              <th>Source</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {alerts.length > 0 ? (
              alerts.map((alert) => (
                <tr key={alert.id}>
                  <td>{alert.type}</td>
                  <td>{alert.location}</td>
                  <td>{alert.source}</td>
                  <td>{alert.timestamp}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4" className="text-center">No alerts found</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AlertHistoryTable;
