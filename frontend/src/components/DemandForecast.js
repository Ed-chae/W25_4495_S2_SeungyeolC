// src/components/DemandForecast.js
import React, { useEffect, useState } from "react";
import api from "../services/api";
import { motion } from "framer-motion";

const DemandForecast = () => {
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
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-2xl font-semibold text-blue-700 mb-4">
        📦 Demand Forecast (Next 7 Days)
      </h2>

      {error && <p className="text-red-500">{error}</p>}

      {!error && forecastData.length > 0 ? (
        <div className="centered-table">
          <table className="min-w-full text-sm border border-gray-200 rounded-md overflow-hidden">
            <thead className="bg-gray-100 text-gray-700">
              <tr>
                <th className="px-4 py-2 border">Item Name</th>
                <th className="px-4 py-2 border">Expected Demand (7 days)</th>
              </tr>
            </thead>
            <tbody>
              {forecastData.map((item, index) => (
                <tr key={index} className="text-center hover:bg-gray-50">
                  <td className="px-4 py-2 border font-medium">{item.product}</td>
                  <td className="px-4 py-2 border">{item.forecast_next_7_days}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !error && <p className="text-gray-500">No forecast data available.</p>
      )}
    </motion.div>
  );
};

export default DemandForecast;
