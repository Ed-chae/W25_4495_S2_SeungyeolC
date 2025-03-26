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
        <div>
            <h1>Business Analytics Dashboard</h1>
            <FileUpload />
            <SentimentChart />
            <RevenueChart />
            <WeatherImpact />
            <CustomerSegments />
            <DemandForecast />
            <SalesAnomalies />
            <Recommendations />
            <MarketBasket />
        </div>
    );
};

export default Dashboard;
