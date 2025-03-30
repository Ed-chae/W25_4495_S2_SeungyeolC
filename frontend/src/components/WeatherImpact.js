import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// ✅ Register Bar chart elements
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const WeatherImpact = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/weather-impact/?city=Vancouver")
      .then((response) => setWeatherData(response.data))
      .catch((err) => {
        console.error("Error fetching weather impact:", err);
        setError("❌ Failed to fetch weather impact.");
      });
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!weatherData) return <p>Loading weather impact...</p>;

  return (
    <div className="p-4">
      <h3 className="text-xl font-bold mb-2">⛅ Weather Impact on Revenue</h3>
      <p><strong>City:</strong> {weatherData.city}</p>
      <p><strong>Temperature:</strong> {weatherData.temperature}°C</p>
      <p><strong>Weather:</strong> {weatherData.weather}</p>
      <p><strong>Predicted Revenue:</strong> ${weatherData.predicted_revenue}</p>

      <Bar
        data={{
          labels: ["Predicted Revenue"],
          datasets: [
            {
              label: "Revenue",
              data: [weatherData.predicted_revenue],
              backgroundColor: "#FF5733",
            },
          ],
        }}
      />
    </div>
  );
};

export default WeatherImpact;
