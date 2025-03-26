import React, { useEffect, useState } from "react";
import api from "../services/api";

function DemandForecast() {
  const [forecast, setForecast] = useState([]);

  useEffect(() => {
    api.get("/demand-forecast/")
      .then((res) => setForecast(res.data))
      .catch((err) => console.error("Error fetching demand forecast:", err));
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📦 Demand Forecast (Next 7 Days)</h2>
      <ul className="space-y-2">
        {forecast.map((item, index) => (
          <li key={index} className="bg-white shadow p-3 rounded">
            <strong>{item.product}</strong> will sell <strong>{item.forecast_next_7_days}</strong> units in the next 7 days.
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DemandForecast;
