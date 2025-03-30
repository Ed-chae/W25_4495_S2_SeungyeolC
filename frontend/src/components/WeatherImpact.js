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

// Register necessary Chart.js components
Chart.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const WeatherImpact = () => {
  const [weatherData, setWeatherData] = useState(null);

  useEffect(() => {
    axios
      .get("/weather-impact/?city=Vancouver")
      .then((response) => setWeatherData(response.data))
      .catch((error) => console.error("Error fetching weather impact:", error));
  }, []);

  if (!weatherData || !Array.isArray(weatherData.forecast)) {
    return <p>Loading weather-based revenue forecast...</p>;
  }

  const chartData = {
    labels: weatherData.forecast.map((entry) => entry.date),
    datasets: [
      {
        label: "Predicted Revenue",
        data: weatherData.forecast.map((entry) => entry.predicted_revenue),
        borderColor: "#4CAF50",
        fill: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: `Weather Impact on Revenue - ${weatherData.city}`,
      },
    },
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">⛅ Weather Impact on Revenue (Next 7 Days)</h2>
      <Line data={chartData} options={chartOptions} />
      <ul className="mt-4 text-sm text-gray-700">
        {weatherData.forecast.map((entry, index) => (
          <li key={index}>
            <b>{entry.date}</b>: {entry.weather}, {entry.temperature}°C, Revenue: ${entry.predicted_revenue}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default WeatherImpact;
