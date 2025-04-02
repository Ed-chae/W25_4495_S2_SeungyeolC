import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Line } from "react-chartjs-2";
import { motion } from "framer-motion";
import {
  Chart,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const WeatherImpact = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios
      .get("/weather-impact/?city=Vancouver")
      .then((res) => setData(res.data))
      .catch((err) => console.error("Error fetching weather impact:", err));
  }, []);

  if (!data) return <p className="text-gray-500">Loading weather impact...</p>;

  const labels = data.forecast.map((d) => d.date);
  const values = data.forecast.map((d) => d.predicted_revenue);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Predicted Revenue",
        data: values,
        fill: false,
        borderColor: "#F59E0B",
        tension: 0.3,
      },
    ],
  };

  return (
    <motion.div
      className="p-4"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-semibold text-yellow-600 mb-2">
        ⛅ Weather Impact on Revenue (Next 7 Days)
      </h2>
      <div className="bg-white rounded shadow p-4">
        <p className="text-sm text-gray-600 mb-2">
          <strong>City:</strong> {data.city} | <strong>Today:</strong>{" "}
          {data.today.weather}, {data.today.temperature}°C
        </p>
        <Line data={chartData} />
      </div>
    </motion.div>
  );
};

export default WeatherImpact;
