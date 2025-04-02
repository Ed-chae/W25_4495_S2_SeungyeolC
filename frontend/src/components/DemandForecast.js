// src/components/DemandForecast.js
import React, { useEffect, useState } from "react";
import api from "../services/api";

function DemandForecast() {
  const [forecastData, setForecastData] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/demand-forecast/")
      .then((res) => {
        if (Array.isArray(res.data.forecast)) {
          setForecastData(res.data.forecast);
        } else {
          setError("Unexpected response format.");
        }
      })
      .catch((err) => {
        console.error("Error fetching demand forecast:", err);
        setError("❌ Failed to fetch demand forecast.");
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📦 Demand Forecast (Next 7 Days)</h2>

      {error && <p className="text-red-600">{error}</p>}

      {forecastData.length > 0 ? (
        <ul className="space-y-2">
          {forecastData.map((item, index) => (
            <li key={index} className="bg-white shadow p-3 rounded">
              <strong>{item.product}</strong>: Forecast {item.forecast_next_7_days} units in 7 days
            </li>
          ))}
        </ul>
      ) : (
        !error && <p>No forecast data available.</p>
      )}
    </div>
  );
}

export default DemandForecast;
