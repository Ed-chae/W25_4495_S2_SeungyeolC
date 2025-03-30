import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Line } from "react-chartjs-2";
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

// ✅ Register required components
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
  const [forecastData, setForecastData] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get("/revenue-forecast/")
      .then((response) => {
        const prophet = response.data.prophet_forecast;
        const lstm = response.data.lstm_forecast;

        if (!prophet?.length && !lstm?.length) {
          setMessage("📭 No forecast data available. Please upload data first.");
          return;
        }

        setForecastData({
          labels: prophet.map((d) => d.ds),
          datasets: [
            {
              label: "Prophet Forecast",
              data: prophet.map((d) => d.yhat),
              borderColor: "#FF5733",
              tension: 0.4,
              fill: false,
            },
            {
              label: "LSTM Forecast",
              data: lstm.map((d) => d.yhat),
              borderColor: "#4CAF50",
              tension: 0.4,
              fill: false,
            },
          ],
        });
        setMessage("");
      })
      .catch((error) => {
        console.error("Error fetching revenue forecast:", error);
        setMessage("❌ Failed to load revenue forecast.");
      });
  }, []);

  return (
    <div className="bg-white shadow-md rounded p-6 mb-6">
      <h2 className="text-xl font-bold mb-4">📈 Revenue Forecast (Next 30 Days)</h2>
      {message ? (
        <p className="text-gray-600">{message}</p>
      ) : forecastData ? (
        <Line
          data={forecastData}
          options={{
            responsive: true,
            plugins: {
              legend: {
                position: "top",
              },
              title: {
                display: true,
                text: "Revenue Forecast using Prophet & LSTM",
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

export default RevenueChart;
