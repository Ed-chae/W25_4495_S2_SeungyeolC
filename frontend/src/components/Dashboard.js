// Filename: Dashboard.js

import React, { useEffect, useState } from "react";
import { fetchAnalytics } from "../services/api";
import { Bar, Line } from "react-chartjs-2";
import Chart from "chart.js/auto";

const Dashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const data = await fetchAnalytics();
        console.log("Analytics Data:", data);
        setAnalytics(data);
      } catch (error) {
        console.error("Error fetching analytics:", error);
      } finally {
        setLoading(false);
      }
    };
    loadAnalytics();
  }, []);

  if (loading) return <p>Loading analytics...</p>;
  if (!analytics || Object.keys(analytics).length === 0) return <p>No data available.</p>;

  // ✅ Ensure all datasets exist before using them
  const revenueTrends = analytics.revenue_trends || [];
  const bestSellers = analytics.best_sellers || [];
  const sentimentAnalysis = analytics.sentiment_analysis || [];
  const weatherImpact = analytics.weather_impact || [];

  // ✅ Ensure data exists before rendering charts
  const hasRevenueData = revenueTrends.length > 0;
  const hasBestSellersData = bestSellers.length > 0;
  const hasSentimentData = sentimentAnalysis.length > 0;
  const hasWeatherImpactData = weatherImpact.length > 0;

  // ✅ Calculate Key Metrics
  const totalRevenue = hasRevenueData
    ? revenueTrends.reduce((sum, item) => sum + item.total_price, 0).toFixed(2)
    : "0.00";

  const bestSellingItem = hasBestSellersData ? bestSellers[0].menu_item : "N/A";

  const avgSentiment = hasSentimentData
    ? (sentimentAnalysis.reduce((sum, item) => sum + item.sentiment, 0) / sentimentAnalysis.length).toFixed(2)
    : "N/A";

  // 📊 Revenue Trends Chart Data
  const revenueTrendsData = hasRevenueData
    ? {
        labels: revenueTrends.map((item) => item.date),
        datasets: [
          {
            label: "Revenue",
            data: revenueTrends.map((item) => item.total_price),
            backgroundColor: "rgba(75,192,192,0.6)",
            borderColor: "rgba(75,192,192,1)",
            fill: true,
            tension: 0.3, // ✅ Smooth line chart
          },
        ],
      }
    : null;

  // 🔥 Best-Selling Menu Items Chart Data
  const bestSellersData = hasBestSellersData
    ? {
        labels: bestSellers.map((item) => item.menu_item),
        datasets: [
          {
            label: "Units Sold",
            data: bestSellers.map((item) => item.quantity),
            backgroundColor: "rgba(255,99,132,0.6)",
          },
        ],
      }
    : null;

  // 😊 Sentiment Analysis Data
  const sentimentData = hasSentimentData
    ? {
        labels: sentimentAnalysis.map((item) => item.menu_item),
        datasets: [
          {
            label: "Sentiment Score",
            data: sentimentAnalysis.map((item) => item.sentiment),
            backgroundColor: "rgba(54,162,235,0.6)",
          },
        ],
      }
    : null;

  // 🌤️ Weather Impact Chart Data
  const weatherImpactData = hasWeatherImpactData
    ? {
        labels: weatherImpact.map((item) => item.weather_condition),
        datasets: [
          {
            label: "Total Revenue",
            data: weatherImpact.map((item) => item.total_price),
            backgroundColor: "rgba(255,206,86,0.6)",
          },
        ],
      }
    : null;

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center" }}>📊 Analytics Dashboard</h2>

      {/* ✅ Key Metrics */}
      <div style={{ display: "flex", justifyContent: "space-around", marginBottom: "20px" }}>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>Total Revenue</h3>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#2c3e50" }}>${totalRevenue}</p>
        </div>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>Best-Selling Item</h3>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#27ae60" }}>{bestSellingItem}</p>
        </div>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>Avg. Sentiment Score</h3>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#e67e22" }}>{avgSentiment}</p>
        </div>
      </div>

      {/* ✅ Charts Section */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        <div>
          <h3>📊 Revenue Trends</h3>
          {hasRevenueData ? <Line data={revenueTrendsData} /> : <p>No revenue data available.</p>}
        </div>

        <div>
          <h3>🔥 Best-Selling Menu Items</h3>
          {hasBestSellersData ? <Bar data={bestSellersData} /> : <p>No sales data available.</p>}
        </div>

        <div>
          <h3>😊 Customer Sentiment Analysis</h3>
          {hasSentimentData ? <Bar data={sentimentData} /> : <p>No sentiment data available.</p>}
        </div>

        <div>
          <h3>🌤️ Weather Impact on Sales</h3>
          {hasWeatherImpactData ? <Bar data={weatherImpactData} /> : <p>No weather impact data available.</p>}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
