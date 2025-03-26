import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { Line } from "react-chartjs-2";  // ✅ Ensure Line is correctly imported
import { Chart, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";

// ✅ Register Chart.js components (required for React-Chart.js)
Chart.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const RevenueChart = () => {
    const [forecastData, setForecastData] = useState({ labels: [], datasets: [] });

    useEffect(() => {
        axios.get("/revenue-forecast/")
            .then(response => {
                const prophetData = response.data.prophet_forecast;
                const lstmData = response.data.lstm_forecast;

                setForecastData({
                    labels: prophetData.map(d => d.ds),
                    datasets: [
                        {
                            label: "Prophet Forecast",
                            data: prophetData.map(d => d.yhat),
                            borderColor: "#FF5733",
                            fill: false,
                        },
                        {
                            label: "LSTM Forecast",
                            data: lstmData.map(d => d.yhat),
                            borderColor: "#4CAF50",
                            fill: false,
                        }
                    ]
                });
            })
            .catch(error => console.error("Error fetching revenue forecast:", error));
    }, []);

    return (
        <div>
            <h3>Revenue Forecast (Next 30 Days)</h3>
            <Line data={forecastData} />
        </div>
    );
};

export default RevenueChart;
