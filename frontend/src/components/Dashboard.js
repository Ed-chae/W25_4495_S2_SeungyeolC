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
      margin:       0.5,
      filename:     'dashboard_report.pdf',
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, scrollY: 0 },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i = 1) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.15, duration: 0.6 },
    }),
  };

  return (
    <div className="bg-gradient-to-br from-slate-100 to-white min-h-screen p-6 space-y-8" ref={dashboardRef}>
      <motion.h1
        className="text-4xl font-bold text-center text-indigo-700 mb-4"
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
      >
        🍽️ Intelligent Business Analytics Dashboard
      </motion.h1>

      <div className="flex justify-end">
        <button
          onClick={handleExport}
          className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition"
        >
          📄 Export Report as PDF
        </button>
      </div>

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
  );
};

export default Dashboard;
