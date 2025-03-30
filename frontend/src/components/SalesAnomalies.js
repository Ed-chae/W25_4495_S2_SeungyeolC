import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Scatter } from "react-chartjs-2";
import {
  Chart,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Title,
  TimeScale,
} from "chart.js";
import "chartjs-adapter-date-fns";

// 📌 Register required components
Chart.register(LinearScale, PointElement, Tooltip, Legend, Title, TimeScale);

const SalesAnomalies = () => {
  const [anomalyData, setAnomalyData] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get("/sales-anomalies/")
      .then((response) => {
        const sales = response.data;

        if (!sales || sales.length === 0) {
          setMessage("📭 No sales data available. Please upload a file first.");
          return;
        }

        const normalSales = sales.filter((s) => s.is_anomaly === "Normal");
        const anomalies = sales.filter((s) => s.is_anomaly === "Anomaly");

        setAnomalyData({
          datasets: [
            {
              label: "Normal Sales",
              data: normalSales.map((s) => ({ x: s.date, y: s.revenue })),
              backgroundColor: "#4CAF50",
              pointRadius: 4,
            },
            {
              label: "Anomalies",
              data: anomalies.map((s) => ({ x: s.date, y: s.revenue })),
              backgroundColor: "#FF5733",
              pointRadius: 6,
            },
          ],
        });

        setMessage("");
      })
      .catch((error) => {
        console.error("Error fetching sales anomalies:", error);
        setMessage("❌ Failed to load anomaly data.");
      });
  }, []);

  return (
    <div className="bg-white p-6 rounded shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">🚨 Sales Anomaly Detection</h2>

      {message ? (
        <p className="text-gray-600">{message}</p>
      ) : anomalyData ? (
        <Scatter
          data={anomalyData}
          options={{
            responsive: true,
            plugins: {
              title: {
                display: true,
                text: "Sales Revenue vs. Time",
              },
              legend: {
                position: "top",
              },
            },
            scales: {
              x: {
                type: "time",
                title: {
                  display: true,
                  text: "Date",
                },
              },
              y: {
                title: {
                  display: true,
                  text: "Revenue",
                },
              },
            },
          }}
        />
      ) : (
        <p className="text-gray-500">⏳ Loading chart data...</p>
      )}
    </div>
  );
};

export default SalesAnomalies;
