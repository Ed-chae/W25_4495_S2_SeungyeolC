import React, { useEffect, useState } from "react";
import api from "../services/api";
import { motion } from "framer-motion";

function DemandForecast() {
  const [forecastData, setForecastData] = useState([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/demand-forecast/")
      .then((res) => {
        if (res.data?.forecast) {
          setForecastData(res.data.forecast);
          setMessage(res.data.message);
        } else {
          setForecastData([]);
          setMessage("No forecast data available.");
        }
      })
      .catch((err) => {
        console.error("Error fetching demand forecast:", err);
        setError("❌ Failed to load demand forecast.");
      });
  }, []);

  return (
    <motion.div
      className="p-4 space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-bold text-indigo-700">
        📦 Demand Forecast (Next 7 Days)
      </h2>

      {error && <p className="text-red-500">{error}</p>}

      {!error && forecastData.length === 0 ? (
        <p className="text-gray-500">{message || "No data available."}</p>
      ) : (
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {forecastData.map((item, idx) => (
            <li
              key={idx}
              className="bg-white rounded shadow p-4 border border-gray-200"
            >
              <p className="font-medium">{item.menu_item}</p>
              <p className="text-sm text-gray-600">
                Forecast:{" "}
                <span className="text-indigo-600 font-semibold">
                  {item.forecast_next_7_days}
                </span>{" "}
                units in 7 days
              </p>
            </li>
          ))}
        </ul>
      )}
    </motion.div>
  );
}

export default DemandForecast;
