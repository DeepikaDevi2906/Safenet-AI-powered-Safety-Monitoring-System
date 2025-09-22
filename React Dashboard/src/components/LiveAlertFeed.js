// LiveAlertFeed.js
import React, { useEffect, useState } from "react";
import socket from '../socket_alerts';
import 'bootstrap/dist/css/bootstrap.min.css';

const LiveAlertFeed = () => {
  const [liveAlerts, setLiveAlerts] = useState([]);

  useEffect(() => {
    socket.on("new_alert", (data) => {
      setLiveAlerts((prev) => [data, ...prev]);
    });
    return () => {
      socket.off("new_alert");
    };
  }, []);

  return (
    <div className="container my-5">
      <h2 className="text-center text-primary mb-4">Live Alerts</h2>
      {liveAlerts.length === 0 ? (
        <div className="alert alert-info text-center">No live alerts yet...</div>
      ) : (
        <ul className="list-group">
          {liveAlerts.map((alert, index) => (
            <li key={index} className="list-group-item d-flex justify-content-between align-items-center">
              <span>
                <strong className="text-danger">{alert.type}</strong> from <em>{alert.source}</em>
              </span>
              <span className="badge bg-secondary">{alert.location || "Unknown Location"}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default LiveAlertFeed;
