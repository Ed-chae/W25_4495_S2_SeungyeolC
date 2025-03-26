import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Bar } from "react-chartjs-2";

const WeatherImpact = () => {
    const [weatherData, setWeatherData] = useState(null);

    useEffect(() => {
        axios.get("/weather-impact/?city=Vancouver")
            .then(response => setWeatherData(response.data))
            .catch(error => console.error("Error fetching weather impact:", error));
    }, []);

    if (!weatherData) return <p>Loading weather impact...</p>;

    return (
        <div>
            <h3>Weather Impact on Revenue</h3>
            <p><b>City:</b> {weatherData.city}</p>
            <p><b>Temperature:</b> {weatherData.temperature}°C</p>
            <p><b>Weather:</b> {weatherData.weather}</p>
            <p><b>Predicted Revenue:</b> ${weatherData.predicted_revenue}</p>

            <Bar
                data={{
                    labels: ["Predicted Revenue"],
                    datasets: [{
                        label: "Revenue",
                        data: [weatherData.predicted_revenue],
                        backgroundColor: "#FF5733"
                    }]
                }}
            />
        </div>
    );
};

export default WeatherImpact;
