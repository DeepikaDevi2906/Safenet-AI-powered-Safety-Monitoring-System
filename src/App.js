import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import AlertHistoryTable from './components/AlertHistoryTable';
import LiveAlertFeed from './components/LiveAlertFeed';
import MapView from './components/MapView';
import StatsPanel from './components/StatsPanel';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';

function App() {
  return (
    <Router>
      <Navbar />
      <div>
        <Routes>
          <Route path="/" element={<HomePage/>} />
          <Route path="/live" element={<LiveAlertFeed />} />
          <Route path="/alerts" element={<AlertHistoryTable />} />
          <Route path="/map" element={<MapView />} />
          <Route path="/stats" element={<StatsPanel />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

