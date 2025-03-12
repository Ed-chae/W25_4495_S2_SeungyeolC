import React, { useState, useEffect } from "react";
import { Line } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { fetchForecast } from "../services/api";

const DashboardForecast = () => {
  const [forecastData, setForecastData] = useState([]);

  useEffect(() => {
    fetchForecast().then((data) => {
      setForecastData(data.prophet_forecast);
    });
  }, []);

  return (
    <Card className="p-4 shadow-lg">
      <h2 className="text-xl font-bold mb-4">Revenue Forecast</h2>
      <LineChart width={600} height={300} data={forecastData}>
        <XAxis dataKey="ds" tickFormatter={(tick) => new Date(tick).toLocaleDateString()} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="yhat" stroke="#8884d8" />
      </LineChart>
    </Card>
  );
};

export default DashboardForecast;
