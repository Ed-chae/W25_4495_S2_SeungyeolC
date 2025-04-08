// src/components/RevenueChart.js
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
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get("/revenue-forecast/")
      .then(response => {
        const prophetData = response.data.prophet_forecast;

        if (!prophetData?.length) {
          setError("No forecast data available.");
          return;
        }

        setForecastData({
          labels: prophetData.map(d => d.ds),
          datasets: [
            {
              label: "📈 Prophet Forecast",
              data: prophetData.map(d => d.yhat),
              borderColor: "#3B82F6",
              backgroundColor: "rgba(59, 130, 246, 0.1)",
              pointRadius: 3,
              tension: 0.4,
              fill: true,
            }
          ]
        });
      })
      .catch(error => {
        console.error("Error fetching revenue forecast:", error);
        setError("❌ Failed to load revenue forecast.");
      });
  }, []);

  return (
    <motion.div
      className="dashboard-card"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-semibold text-blue-700 mb-2">📊 Revenue Forecast (Next 30 Days)</h2>
      <p className="text-sm text-gray-600 mb-4">
        This chart shows the expected revenue based on past trends using Prophet.
      </p>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {!error && forecastData.labels.length > 0 && (
        <div className="overflow-x-auto">
          <Line
            data={forecastData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: "top" },
                tooltip: { mode: "index", intersect: false },
              },
              scales: {
                y: {
                  title: {
                    display: true,
                    text: "Revenue ($)"
                  }
                },
                x: {
                  ticks: {
                    maxTicksLimit: 10
                  }
                }
              }
            }}
          />
        </div>
      )}
    </motion.div>
  );
};

export default RevenueChart;
