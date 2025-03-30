import React, { useEffect, useState } from "react";
import api from "../services/api";

function DemandForecast() {
  const [forecast, setForecast] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    api.get("/demand-forecast/")
      .then((res) => {
        if (res.data && Array.isArray(res.data.forecast)) {
          setForecast(res.data.forecast);
          setMessage(res.data.message || "");
        } else {
          setForecast([]);
          setMessage("⚠️ Unexpected data format received.");
        }
      })
      .catch((err) => {
        console.error("Error fetching demand forecast:", err);
        setMessage("❌ Failed to load demand forecast.");
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📦 Demand Forecast (Next 7 Days)</h2>

      {message && <p className="text-sm text-gray-600 mb-2">{message}</p>}

      {forecast.length === 0 ? (
        <p className="text-gray-500">No forecast data available.</p>
      ) : (
        <ul className="space-y-2">
          {forecast.map((item, index) => (
            <li key={index} className="bg-white shadow p-3 rounded">
              <strong>{item.menu_item}</strong> is expected to sell <strong>{item.forecast_next_7_days}</strong> units in the next 7 days.
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default DemandForecast;
