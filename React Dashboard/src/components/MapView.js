import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import axios from "axios";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import 'bootstrap/dist/css/bootstrap.min.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const MapView = () => {
  const [locations, setLocations] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/alerts")
      .then((res) => {
        const locs = res.data
          .map((alert) => {
            const parts = alert.location.split(",");
            if (parts.length === 2) {
              return {
                lat: parseFloat(parts[0]),
                lng: parseFloat(parts[1]),
                type: alert.type,
                source: alert.source,
                timestamp: alert.timestamp,
              };
            }
            return null;
          })
          .filter(Boolean);
        setLocations(locs);
      })
      .catch((err) => {
        console.error("Error fetching alert locations:", err);
      });
  }, []);

  return (
    <div className="container my-5">
      <div className="card shadow">
        <div className="card-body">
          <h2 className="card-title text-center text-success mb-4">Alert Location Map</h2>
          <MapContainer
            center={[10.7905, 78.7047]}
            zoom={12}
            style={{ height: "400px", width: "100%" }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {locations.map((loc, index) => (
              <Marker key={index} position={[loc.lat, loc.lng]}>
                <Popup>
                  <strong>{loc.type}</strong><br />
                  From: {loc.source}<br />
                  Time: {loc.timestamp}
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </div>
    </div>
  );
};

export default MapView;
