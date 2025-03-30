import React from "react";
import FileUpload from "./FileUpload";
import SentimentChart from "./SentimentChart";
import RevenueChart from "./RevenueChart";
import WeatherImpact from "./WeatherImpact";
import CustomerSegments from "./CustomerSegments";
import DemandForecast from "./DemandForecast";
import SalesAnomalies from "./SalesAnomalies";
import Recommendations from "./Recommendations";
import MarketBasket from "./MarketBasket";

const Dashboard = () => {
  return (
    <div className="bg-gray-100 min-h-screen p-6 space-y-8">
      <h1 className="text-3xl font-bold text-center text-blue-700 mb-4">
        🍽️ Business Analytics Dashboard
      </h1>

      <div className="bg-white p-4 rounded shadow">
        <FileUpload />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <SentimentChart />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <RevenueChart />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <WeatherImpact />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <CustomerSegments />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <DemandForecast />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <SalesAnomalies />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <Recommendations />
      </div>

      <div className="bg-white p-4 rounded shadow">
        <MarketBasket />
      </div>
    </div>
  );
};

export default Dashboard;
