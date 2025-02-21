// Filename: Dashboard.js

import React, { useEffect, useState } from "react";
import { fetchAnalytics } from "../services/api";
import { Bar, Line, Pie } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { CategoryScale, LinearScale } from "chart.js";
import { ArcElement } from "chart.js";
import { Chart as ChartJS } from "chart.js";

// ✅ Register necessary scales & elements for Chart.js
ChartJS.register(CategoryScale, LinearScale, ArcElement);

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

  // ✅ Extracting data safely
  const revenueTrends = analytics.revenue_trends || [];
  const bestSellers = analytics.best_sellers || [];
  const sentimentStats = analytics.sentiment_stats || [];
  const weatherImpact = analytics.weather_impact || [];
  const futureRevenue = analytics.future_revenue || [];
  const bestMenu = analytics.best_menu || {};
  const worstMenu = analytics.worst_menu || {};

  // ✅ Ensure data exists before rendering
  const hasRevenueData = revenueTrends.length > 0;
  const hasBestSellersData = bestSellers.length > 0;
  const hasSentimentData = sentimentStats.length > 0;
  const hasWeatherImpactData = weatherImpact.length > 0;
  const hasFutureRevenueData = futureRevenue.length > 0;

  // ✅ Sort sentiment stats by highest positive review percentage
  const sortedSentimentStats = [...sentimentStats].sort((a, b) => b.positive_review_percentage - a.positive_review_percentage);

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center" }}>📊 Analytics Dashboard</h2>

      {/* ✅ Key Metrics */}
      <div style={{ display: "flex", justifyContent: "space-around", marginBottom: "20px" }}>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>Total Revenue</h3>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#2c3e50" }}>${revenueTrends.reduce((sum, item) => sum + item.total_price, 0).toFixed(2)}</p>
        </div>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>🏆 Best Menu Item</h3>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#27ae60" }}>
            {bestMenu.menu_item || "N/A"} ({bestMenu.positive_review_percentage ? `${bestMenu.positive_review_percentage}%` : "0%"} positive)
          </p>
        </div>
        <div style={{ padding: "10px", backgroundColor: "#f3f4f6", borderRadius: "8px", textAlign: "center" }}>
          <h3>🥶 Worst Menu Item</h3>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#e74c3c" }}>
            {worstMenu.menu_item || "N/A"} ({worstMenu.negative_review_percentage ? `${worstMenu.negative_review_percentage}%` : "0%"} negative)
          </p>
        </div>
      </div>

      {/* 😊 Sorted Sentiment Analysis Table */}
      <div style={{ marginBottom: "20px" }}>
        <h3>😊 Customer Sentiment Analysis</h3>
        {hasSentimentData ? (
          <table border="1" cellPadding="5" style={{ width: "100%", textAlign: "center" }}>
            <thead>
              <tr style={{ backgroundColor: "#f3f4f6" }}>
                <th>Menu Item</th>
                <th>Positive (%)</th>
                <th>Negative (%)</th>
              </tr>
            </thead>
            <tbody>
              {sortedSentimentStats.map((item, index) => (
                <tr key={index}>
                  <td>{item.menu_item}</td>
                  <td>{item.positive_review_percentage || "0"}%</td>
                  <td>{item.negative_review_percentage || "0"}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p>No sentiment data available.</p>}
      </div>

      {/* 📈 Future Revenue Prediction Table */}
      <div style={{ marginBottom: "20px" }}>
        <h3>📈 Future Revenue Prediction (Next Week)</h3>
        {hasFutureRevenueData ? (
          <table border="1" cellPadding="5" style={{ width: "100%", textAlign: "center" }}>
            <thead>
              <tr style={{ backgroundColor: "#f3f4f6" }}>
                <th>Date</th>
                <th>Weather</th>
                <th>Predicted Revenue ($)</th>
              </tr>
            </thead>
            <tbody>
              {futureRevenue.map((item, index) => (
                <tr key={index}>
                  <td>{item.date}</td>
                  <td>{item.weather_condition}</td>
                  <td>${item.predicted_revenue.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p>No prediction data available.</p>}
      </div>

      {/* 📊 Revenue Trends */}
      <div style={{ marginBottom: "20px" }}>
        <h3>📊 Revenue Trends</h3>
        {hasRevenueData ? (
          <Line
            data={{
              labels: revenueTrends.map((item) => item.date),
              datasets: [
                {
                  label: "Revenue",
                  data: revenueTrends.map((item) => item.total_price),
                  backgroundColor: "rgba(75,192,192,0.6)",
                  borderColor: "rgba(75,192,192,1)",
                  fill: true,
                  tension: 0.3,
                },
              ],
            }}
          />
        ) : (
          <p>No revenue data available.</p>
        )}
      </div>
      
      {/* 🔥 Best-Selling Items Pie Chart */}
      <div style={{ marginBottom: "20px" }}>
        <h3>🔥 Best-Selling Menu Items</h3>
        {hasBestSellersData ? (
          <Pie
            data={{
              labels: bestSellers.map((item) => item.menu_item),
              datasets: [
                {
                  data: bestSellers.map((item) => item.quantity),
                  backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#8A2BE2", "#FF4500"],
                },
              ],
            }}
          />
        ) : (
          <p>No sales data available.</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
