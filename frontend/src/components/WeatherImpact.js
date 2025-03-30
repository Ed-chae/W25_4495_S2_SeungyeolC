import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Bar } from "react-chartjs-2";

const WeatherImpact = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/weather-impact/?city=Vancouver")
      .then((response) => {
        setWeatherData(response.data);
        setError("");
      })
      .catch((error) => {
        console.error("Error fetching weather impact:", error);
        setError("❌ Failed to load weather impact data.");
      });
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!weatherData) return <p>Loading weather impact...</p>;

  return (
    <div className="p-4 bg-white shadow-md rounded mb-6">
      <h2 className="text-xl font-semibold mb-4">⛅ Weather Impact on Revenue</h2>
      <div className="mb-4 text-sm text-gray-700">
        <p><strong>City:</strong> {weatherData.city}</p>
        <p><strong>Temperature:</strong> {weatherData.temperature}°C</p>
        <p><strong>Humidity:</strong> {weatherData.humidity}%</p>
        <p><strong>Weather:</strong> {weatherData.weather}</p>
        <p><strong>Predicted Revenue:</strong> ${weatherData.predicted_revenue}</p>
      </div>

      <Bar
        data={{
          labels: ["Predicted Revenue"],
          datasets: [
            {
              label: "Revenue ($)",
              data: [weatherData.predicted_revenue],
              backgroundColor: "#4B9CD3"
            }
          ]
        }}
        options={{
          responsive: true,
          plugins: {
            legend: {
              position: "top"
            },
            title: {
              display: true,
              text: "Impact of Weather on Expected Revenue"
            }
          },
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }}
      />
    </div>
  );
};

export default WeatherImpact;
