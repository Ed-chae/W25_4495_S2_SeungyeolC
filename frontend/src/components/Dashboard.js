// src/components/Dashboard.js
import React, { useRef } from "react";
import { motion } from "framer-motion";
import html2pdf from "html2pdf.js";
import FileUpload from "./FileUpload";
import ResetButton from "./ResetButton";
import SentimentChart from "./SentimentChart";
import RevenueChart from "./RevenueChart";
import WeatherImpact from "./WeatherImpact";
import CustomerSegments from "./CustomerSegments";
import DemandForecast from "./DemandForecast";
import Recommendations from "./Recommendations";
import MarketBasket from "./MarketBasket";
import MenuCategoryChart from "./MenuCategoryChart";

const Dashboard = () => {
  const dashboardRef = useRef();

  const handleExport = () => {
    const element = dashboardRef.current;
    const opt = {
      margin: 0.5,
      filename: "dashboard_report.pdf",
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, scrollY: 0 },
      jsPDF: { unit: "in", format: "letter", orientation: "portrait" },
    };
    html2pdf().set(opt).from(element).save();
  };

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i = 1) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.1, duration: 0.6 },
    }),
  };

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-emerald-400 to-emerald-500 p-6"
      ref={dashboardRef}
    >
      <div className="max-w-6xl mx-auto bg-white rounded-2xl p-6 shadow-lg space-y-8">
        {/* Header */}
        <motion.h1
          className="text-4xl font-bold text-center text-gray-800 mb-2"
          initial="hidden"
          animate="visible"
          variants={fadeInUp}
        >
          🍽️ Intelligent Business Analytics Dashboard
        </motion.h1>

        {/* Export Button */}
        <div className="card">
          <button
            onClick={handleExport}
            className="bg-indigo-600 text-white px-4 py-2 rounded-md shadow hover:bg-indigo-700 transition"
          >
            📄 Export Report as PDF
          </button>
        </div>

        {/* Upload & Reset Section */}
        <motion.div
          className="bg-gray-50 p-4 rounded-xl shadow-md"
          initial="hidden"
          animate="visible"
          variants={fadeInUp}
        >
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <FileUpload />
            <ResetButton />
          </div>
        </motion.div>

        {/* Dashboard Widgets */}
        {[
          <SentimentChart />,
          <MenuCategoryChart />,
          <RevenueChart />,
          <WeatherImpact />,
          <CustomerSegments />,
          <DemandForecast />,
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
    </div>
  );
};

export default Dashboard;
