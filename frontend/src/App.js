import React from "react";
import FileUpload from "./components/FileUpload";
import SentimentChart from "./components/SentimentChart";
import RevenueChart from "./components/RevenueChart";
import WeatherImpact from "./components/WeatherImpact";
import CustomerSegments from "./components/CustomerSegments";
import DemandForecast from "./components/DemandForecast";
import Recommendations from "./components/Recommendations";
import MarketBasket from "./components/MarketBasket";
import ResetButton from "./components/ResetButton";
import "./styles.css";

function App() {
    return (
        <div>
            <h1>Business Analytics Dashboard</h1>
            <FileUpload />
            <ResetButton />
            <SentimentChart />
            <RevenueChart />
            <WeatherImpact />
            <CustomerSegments />
            <DemandForecast />
            <Recommendations />
            <MarketBasket />
        </div>
    );
}

export default App;
