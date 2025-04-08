// src/components/WeatherImpact.js
import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Line } from "react-chartjs-2";
import {
  Chart,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { motion } from "framer-motion";

Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend);

const WeatherImpact = () => {
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/weather-impact/?city=Vancouver")
      .then((response) => setForecastData(response.data))
      .catch((err) => {
        console.error("Error fetching weather impact:", err);
        setError("❌ Failed to load weather forecast.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!forecastData || !forecastData.forecast) return <p>Loading weather forecast...</p>;

  const data = {
    labels: forecastData.forecast.map((d) => d.date),
    datasets: [
      {
        label: "Predicted Revenue",
        data: forecastData.forecast.map((d) => d.predicted_revenue),
        borderColor: "#4F46E5",
        backgroundColor: "#c7d2fe",
        tension: 0.4,
        fill: true,
        pointRadius: 3,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { callback: (val) => `$${val}` },
      },
    },
  };

  return (
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-2xl font-bold text-indigo-700 mb-3">
        🌤️ Weather Impact on Revenue
      </h2>

      <p className="text-sm text-gray-600 mb-4">
        City: <strong>{forecastData.city}</strong> | Today’s Weather:{" "}
        <strong>{forecastData.forecast[0].weather}</strong> | Temp:{" "}
        <strong>{forecastData.forecast[0].temperature}°C</strong>
      </p>

      <div className="h-64">
        <Line data={data} options={options} />
      </div>
    </motion.div>
  );
};

export default WeatherImpact;
