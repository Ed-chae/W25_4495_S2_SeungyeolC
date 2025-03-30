import React, { useEffect, useState } from "react";
import api from "../services/api";

function DemandForecast() {
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/demand-forecast/")
      .then((res) => {
        setForecast(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching demand forecast:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <p className="p-4">Loading demand forecast...</p>;
  }

  if (forecast.length === 0) {
    return (
      <div className="p-4">
        <h2 className="text-xl font-bold mb-4">📦 Demand Forecast</h2>
        <p className="text-gray-600">No forecast data available. Please upload restaurant order data to see results.</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📦 Demand Forecast (Next 7 Days)</h2>
      <ul className="space-y-2">
        {forecast.map((item, index) => (
          <li key={index} className="bg-white shadow p-3 rounded">
            <strong>{item.menu_item}</strong> is expected to sell <strong>{item.forecast_next_7_days}</strong> units.
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DemandForecast;
