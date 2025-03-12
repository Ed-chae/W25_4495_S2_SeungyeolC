// Filename: Dashboard.js

import React, { useState, useEffect } from "react";
import { fetchAnalytics, fetchForecast, uploadFile } from "../services/api";
import { Bar, Line, Pie } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { CategoryScale, LinearScale, ArcElement } from "chart.js";

Chart.register(CategoryScale, LinearScale, ArcElement);

const Dashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [forecastData, setForecastData] = useState([]);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const data = await fetchAnalytics();
        if (data && data.salesData && data.customerData) {
          setAnalytics(data);
        } else {
          console.warn("Analytics API returned no data.");
        }
      } catch (error) {
        console.warn("Skipping analytics due to missing API.");
      } finally {
        setLoading(false);
      }
    };

    const loadForecast = async () => {
      try {
        const forecast = await fetchForecast();
        if (forecast.prophet_forecast && forecast.prophet_forecast.length > 0) {
          setForecastData(forecast.prophet_forecast);
        } else {
          console.warn("Forecast API returned no data.");
        }
      } catch (error) {
        console.error("Error fetching forecast:", error);
      }
    };

    loadAnalytics();
    loadForecast();
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (file) {
      console.log("Uploading file: ", file.name);
      const response = await uploadFile(file);
      if (!response.error) {
        setForecastData(await fetchForecast());  // ✅ Refresh AI forecast after upload
      }
    }
  };

  if (loading) return <p>Loading analytics...</p>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Dashboard</h1>

      {/* ✅ File Upload Section */}
      <div className="mb-4">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} className="ml-2 p-2 bg-blue-500 text-white rounded">
          Upload & Update Forecast
        </button>
      </div>

      {/* ✅ Sales Analytics & Customer Insights */}
      {analytics && (
        <>
          <div className="mb-6">
            <h2 className="text-xl font-bold mb-2">Sales Analytics</h2>
            <Bar data={analytics.salesData} options={{ responsive: true }} />
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-bold mb-2">Customer Insights</h2>
            <Pie data={analytics.customerData} options={{ responsive: true }} />
          </div>
        </>
      )}

      {/* ✅ AI Revenue Forecast */}
      <div className="mb-6">
        <h2 className="text-xl font-bold mb-2">AI Revenue Forecast</h2>
        {forecastData.length > 0 ? (
          <Line
            data={{
              labels: forecastData.map((item) => new Date(item.ds).toLocaleDateString()),
              datasets: [
                {
                  label: "Predicted Revenue",
                  data: forecastData.map((item) => item.yhat),
                  borderColor: "#4CAF50",
                  backgroundColor: "rgba(76, 175, 80, 0.2)",
                  tension: 0.4,
                },
              ],
            }}
            options={{ responsive: true }}
          />
        ) : (
          <p>No forecast data available</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
