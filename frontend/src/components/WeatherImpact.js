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

Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend);

const WeatherImpact = () => {
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get("/weather-impact/?city=Vancouver")
      .then(response => {
        setForecastData(response.data);
      })
      .catch(error => {
        console.error("Error fetching weather impact:", error);
        setError("Failed to load weather forecast.");
      });
  }, []);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!forecastData || !forecastData.forecast) return <p>Loading weather forecast...</p>;

  const data = {
    labels: forecastData.forecast.map(d => d.date),
    datasets: [
      {
        label: "Predicted Revenue",
        data: forecastData.forecast.map(d => d.predicted_revenue),
        borderColor: "#4F46E5",
        tension: 0.3,
        fill: false,
      },
    ],
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-2">🌤️ Weather Impact on Revenue (7-Day Forecast)</h2>
      <p className="text-gray-600 mb-4">
        City: <strong>{forecastData.city}</strong>, 
        Weather (today): <strong>{forecastData.forecast[0].weather}</strong>, 
        Temp: <strong>{forecastData.forecast[0].temperature}°C</strong>
      </p>
      <Line data={data} />
    </div>
  );
};

export default WeatherImpact;
