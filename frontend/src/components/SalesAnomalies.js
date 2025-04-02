import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Scatter } from "react-chartjs-2";
import { Chart as ChartJS, TimeScale, LinearScale, PointElement, Tooltip, Legend } from "chart.js";
import "chartjs-adapter-date-fns";
import { motion } from "framer-motion";

// Register necessary chart.js components
ChartJS.register(TimeScale, LinearScale, PointElement, Tooltip, Legend);

const SalesAnomalies = () => {
  const [anomalyData, setAnomalyData] = useState({ datasets: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get("/sales-anomalies/")
      .then(response => {
        const sales = response.data;

        const normalSales = sales.filter(s => s.is_anomaly === "Normal");
        const anomalies = sales.filter(s => s.is_anomaly === "Anomaly");

        setAnomalyData({
          datasets: [
            {
              label: "Normal Sales",
              data: normalSales.map(s => ({ x: s.date, y: s.revenue })),
              backgroundColor: "#4CAF50",
              pointRadius: 5
            },
            {
              label: "Anomalies",
              data: anomalies.map(s => ({ x: s.date, y: s.revenue })),
              backgroundColor: "#FF5733",
              pointRadius: 6
            }
          ]
        });
      })
      .catch(error => console.error("Error fetching sales anomalies:", error))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-gray-500">Loading anomaly chart...</p>;

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top"
      },
      tooltip: {
        callbacks: {
          label: (context) => `Revenue: $${context.parsed.y.toFixed(2)}`
        }
      }
    },
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day"
        },
        title: {
          display: true,
          text: "Date"
        }
      },
      y: {
        title: {
          display: true,
          text: "Revenue ($)"
        }
      }
    }
  };

  return (
    <motion.div
      className="p-4"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h2 className="text-xl font-semibold text-red-600 mb-4">🚨 Sales Anomaly Detection</h2>
      <Scatter data={anomalyData} options={options} />
    </motion.div>
  );
};

export default SalesAnomalies;
