import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Scatter } from "react-chartjs-2";
import { Chart, LinearScale, PointElement, Title, Tooltip, Legend, TimeScale } from "chart.js";
import 'chartjs-adapter-date-fns';

// Register required chart.js components
Chart.register(LinearScale, PointElement, Title, Tooltip, Legend, TimeScale);

const SalesAnomalies = () => {
  const [anomalyData, setAnomalyData] = useState({ datasets: [] });
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios.get("/sales-anomalies/")
      .then(response => {
        const sales = response.data;

        if (!Array.isArray(sales)) {
          setMessage("⚠️ Invalid data format received.");
          return;
        }

        const normalSales = sales.filter(s => s.is_anomaly === "Normal");
        const anomalies = sales.filter(s => s.is_anomaly === "Anomaly");

        setAnomalyData({
          datasets: [
            {
              label: "Normal Sales",
              data: normalSales.map(s => ({ x: s.date, y: s.quantity })),
              backgroundColor: "#4CAF50"
            },
            {
              label: "Anomalies",
              data: anomalies.map(s => ({ x: s.date, y: s.quantity })),
              backgroundColor: "#FF5733"
            }
          ]
        });
        setMessage("");
      })
      .catch(error => {
        console.error("Error fetching sales anomalies:", error);
        setMessage("❌ Failed to load anomaly data.");
      });
  }, []);

  return (
    <div className="p-4">
      <h3 className="text-xl font-bold mb-4">🚨 Sales Anomaly Detection</h3>

      {message && <p className="text-red-600">{message}</p>}

      <Scatter
        data={anomalyData}
        options={{
          responsive: true,
          plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Sales Anomalies by Date' }
          },
          scales: {
            x: {
              type: "time",
              title: { display: true, text: "Date" }
            },
            y: {
              title: { display: true, text: "Quantity" }
            }
          }
        }}
      />
    </div>
  );
};

export default SalesAnomalies;
