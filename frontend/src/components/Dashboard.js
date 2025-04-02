// src/components/Dashboard.js
import React from "react";
import { motion } from "framer-motion";
import FileUpload from "./FileUpload";
import ResetButton from "./ResetButton";
import SentimentChart from "./SentimentChart";
import RevenueChart from "./RevenueChart";
import WeatherImpact from "./WeatherImpact";
import CustomerSegments from "./CustomerSegments";
import DemandForecast from "./DemandForecast";
import SalesAnomalies from "./SalesAnomalies";
import Recommendations from "./Recommendations";
import MarketBasket from "./MarketBasket";

const Dashboard = () => {
  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i = 1) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.15, duration: 0.6 },
    }),
  };

  return (
    <div className="bg-gradient-to-br from-slate-100 to-white min-h-screen p-6 space-y-8">
      <motion.h1
        className="text-4xl font-bold text-center text-indigo-700 mb-8"
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
      >
        🍽️ Intelligent Business Analytics Dashboard
      </motion.h1>

      <motion.div
        className="bg-white p-4 rounded-xl shadow-md"
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
      >
        <FileUpload />
        <ResetButton />
      </motion.div>

      {[
        <SentimentChart />,
        <RevenueChart />,
        <WeatherImpact />,
        <CustomerSegments />,
        <DemandForecast />,
        <SalesAnomalies />,
        <Recommendations userId={1} />,
        <MarketBasket />,
      ].map((Component, index) => (
        <motion.div
          key={index}
          className="bg-white p-4 rounded-xl shadow-md"
          custom={index + 2}
          initial="hidden"
          animate="visible"
          variants={fadeInUp}
        >
          {Component}
        </motion.div>
      ))}
    </div>
  );
};

export default Dashboard;
