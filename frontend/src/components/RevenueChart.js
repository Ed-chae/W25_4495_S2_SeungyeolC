import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Line } from "react-chartjs-2";
import { motion } from "framer-motion";
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const RevenueChart = () => {
  const [forecastData, setForecastData] = useState({ labels: [], datasets: [] });

  useEffect(() => {
    axios.get("/revenue-forecast/")
      .then(response => {
        const prophetData = response.data.prophet_forecast;
        const lstmData = response.data.lstm_forecast;

        setForecastData({
          labels: prophetData.map(d => d.ds),
          datasets: [
            {
              label: "Prophet Forecast",
              data: prophetData.map(d => d.yhat),
              borderColor: "#3B82F6",
              tension: 0.4,
              fill: false,
            },
            {
              label: "LSTM Forecast",
              data: lstmData.map(d => d.yhat),
              borderColor: "#10B981",
              tension: 0.4,
              fill: false,
            }
          ]
        });
      })
      .catch(error => console.error("Error fetching revenue forecast:", error));
  }, []);

  return (
    <motion.div
      className="p-4"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-semibold mb-4 text-blue-700">📊 Revenue Forecast (Next 30 Days)</h2>
      <div className="bg-white rounded shadow p-4">
        <Line data={forecastData} />
      </div>
    </motion.div>
  );
};

export default RevenueChart;
